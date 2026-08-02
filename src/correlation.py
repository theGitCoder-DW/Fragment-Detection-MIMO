"""
pilot_reference.py

Generates a GFSK reference (template) waveform for the TS-UNB pilot
sequence "011101000010", for use as a matched filter to detect burst
timing by correlating against a received IQ stream.

Modulation parameters are ported directly from FileTx.h (TsUnbLib::Trx)
so the reference shape matches what the real transmitter produces:
    - Binary FSK with deviation = BASE_MULTIPLIER * 39
    - Gaussian pulse shaping, BT = 1
    - OVERSAMPLING = 96 samples/symbol

For a *reference* waveform we generate it at baseband (carrier offset
= 0), i.e. we skip burstFreq/centerFrequency cancellation in the C++
code and just integrate the filtered frequency deviation directly.
This gives a complex baseband template you can correlate against a
received signal that has already been downconverted to baseband.
"""

import numpy as np


# ---- Constants ported from FileTx.h -----------------------------------
BASE_MULTIPLIER = 15.2587890625
OVERSAMPLING = 96
DEVIATION = BASE_MULTIPLIER * 39      # frequency deviation, same units as C++
SYMBOL_RATE_UNITS = BASE_MULTIPLIER * 156   # "samplingRate" in FileTx.h
BT = 1.0                               # Gaussian filter bandwidth-time product

PILOT_BITS = "011101000010"


def bits_to_array(bit_string):
    """'011101000010' -> np.array([0,1,1,1,0,1,0,0,0,0,1,0], dtype=float32)"""
    return np.array([int(b) for b in bit_string], dtype=np.float32)


def gaussian_filter_taps(oversampling, bt):
    """
    Same Gaussian filter construction as FileTx::transmit(), where
    GaussFilter is std::vector<float>:
        N = oversampling / BT
        g[i] = sqrt(2*pi/ln2) * BT * Ta * exp(-2*(pi*BT*t)^2/ln2)
    normalized to unit sum. Computed in float32 to match the C++ type.
    """
    N = int(oversampling / bt)
    n = np.arange(-N, N + 1, dtype=np.float32)
    Ta = np.float32(1.0 / oversampling)
    t = (n / oversampling).astype(np.float32)
    g = (np.sqrt(np.float32(2 * np.pi) / np.float32(np.log(2))) * np.float32(bt) * Ta
         * np.exp(-2 * (np.float32(np.pi) * np.float32(bt) * t) ** 2 / np.float32(np.log(2))))
    g = g.astype(np.float32)
    g /= g.sum()
    return g.astype(np.float32)


def gfsk_modulate(bits, oversampling=OVERSAMPLING, deviation=DEVIATION,
                   symbol_rate_units=SYMBOL_RATE_UNITS, bt=BT,
                   random_phase=False, amplitude=1.0):
    """
    Reproduces the per-burst modulation loop in FileTx::transmit(),
    but at baseband (no carrier / center-frequency term) and without
    the burst-scheduling (delay/tail/power-ramp) bookkeeping — those
    are irrelevant for a correlation template.

    Dtypes match FileTx.h: RadioBurstMod / GaussFilter / BtFiltered are
    std::vector<float> (float32) in the C++, so the modulation and
    filtering here stay in float32 too. Only the phase accumulator
    (signalPhase) is float64, matching the C++ `double signalPhase` --
    that's intentional in the original code, since integrating phase
    over many samples in float32 accumulates rounding error. The final
    IQ output is complex64, matching std::complex<float> IQData.

    Returns
    -------
    iq : np.ndarray[complex64]
        Oversampled complex baseband GFSK waveform for `bits`.
    """
    bits = np.asarray(bits, dtype=np.float32)

    # 1) Bit -> square-wave frequency deviation (same as C++ readBit loop)
    deviation = np.float32(deviation)
    mod = np.where(bits > 0.5, deviation, -deviation).astype(np.float32)
    mod_oversampled = np.repeat(mod, oversampling).astype(np.float32)

    # 2) Gaussian pulse shaping (same filter + edge handling as C++:
    #    clamp to first/last sample outside the array, i.e. 'edge' padding)
    taps = gaussian_filter_taps(oversampling, bt)
    pad = len(taps) // 2
    padded = np.pad(mod_oversampled, (pad, pad), mode="edge").astype(np.float32)
    filtered = np.convolve(padded, taps, mode="valid").astype(np.float32)
    # 'valid' with symmetric padding reproduces the C++ correlation loop
    filtered = filtered[: len(mod_oversampled)]

    # 3) Phase integration -> complex exponential (same recurrence as C++).
    #    Accumulate in float64 (matches C++ `double signalPhase`), then
    #    cast the final samples down to complex64 (matches std::complex<float>).
    phase = np.random.uniform(0, 2 * np.pi) if random_phase else 0.0
    phase_inc = (filtered.astype(np.float64) * (2 * np.pi)
                 / (oversampling * np.float64(symbol_rate_units)))
    phase_track = phase + np.cumsum(phase_inc)
    iq = (np.float32(amplitude) * np.exp(1j * phase_track)).astype(np.complex64)

    return iq



def generate_pilot_reference(pilot_bits=PILOT_BITS, oversampling=OVERSAMPLING):
    """Convenience wrapper: build the reference waveform for the pilot."""
    bits = bits_to_array(pilot_bits)
    return gfsk_modulate(bits, oversampling=oversampling)


# ---- Matched-filter / correlation detector -----------------------------
def correlate_for_burst(received_iq, reference_iq):
    """
    Matched filter: correlate received baseband IQ against the
    (conjugated, time-reversed) pilot reference. Peaks in the
    magnitude output mark candidate burst start positions.

    Returns
    -------
    corr_mag : np.ndarray
        Magnitude of the correlation, same convention as
        np.correlate(received, reference, mode="valid").
    """
    # np.correlate(a, v)[n] = sum_k a[n+k] * conj(v[k]), which is exactly
    # the matched-filter cross-correlation we want (mode="valid" only
    # returns positions where the full reference overlaps `received_iq`).
    corr = np.correlate(received_iq, reference_iq, mode="valid")
    # Normalize by reference energy so the peak height is comparable
    # across different burst amplitudes
    ref_energy = np.sum(np.abs(reference_iq) ** 2)
    corr_mag = np.abs(corr) / np.sqrt(ref_energy)
    return corr_mag


def find_burst_start(received_iq, reference_iq, threshold=None):
    """
    Returns the sample index (into received_iq) of the best correlation
    peak, and its normalized magnitude. If `threshold` is given, returns
    None if the peak doesn't clear it (i.e. no burst detected).
    """
    corr_mag = correlate_for_burst(received_iq, reference_iq)
    peak_idx = int(np.argmax(corr_mag))
    peak_val = float(corr_mag[peak_idx])

    if threshold is not None and peak_val < threshold:
        return None, peak_val

    # With np.correlate(received, reference, mode="valid"), corr_mag[n]
    # is the correlation for the window received[n : n+len(reference)],
    # so the peak index IS the burst start sample directly.
    burst_start = peak_idx
    return burst_start, peak_val


def load_iq_file(path, dtype=np.complex64):
    """
    Load a raw interleaved-float IQ capture, same binary layout as
    FileTx writes: OutputFile.write((char*)IQData.data(),
                                     sizeof(std::complex<float>) * IQData.size())
    i.e. no header, just back-to-back (I, Q) float32 pairs.
    """
    return np.fromfile(path, dtype=dtype)


def find_all_bursts(received_iq, reference_iq, threshold, min_spacing=None):
    """
    Find every correlation peak above `threshold` -- a full mioty
    telegram is split across ~20 uplink bursts, each of which starts
    with this same pilot sequence.

    Uses scipy.signal.find_peaks for proper local-maxima detection.
    (An earlier version of this function grouped "candidate" samples
    by contiguity, which silently collapsed into a single reported
    peak whenever most/all samples passed the threshold -- e.g. at
    threshold=0 -- because the whole array became one contiguous
    group. find_peaks doesn't have that failure mode: it finds true
    local maxima independent of how many samples clear the threshold.)

    min_spacing: minimum sample distance between reported peaks, so
    the mainlobe of one burst's correlation peak isn't reported as
    multiple detections. Defaults to len(reference_iq).

    Returns a list of (sample_index, peak_magnitude) tuples, sorted by
    sample index.
    """
    from scipy.signal import find_peaks

    if min_spacing is None:
        min_spacing = len(reference_iq)

    corr_mag = correlate_for_burst(received_iq, reference_iq)

    peak_indices, _ = find_peaks(corr_mag, height=threshold, distance=min_spacing)
    return [(int(idx), float(corr_mag[idx])) for idx in peak_indices]


def correlation_diagnostics(received_iq, reference_iq):
    """
    Print summary stats of the raw (un-thresholded) correlation
    magnitude, so you can pick a sensible --threshold for YOUR
    capture's amplitude scale instead of guessing. Real captures
    rarely have unit amplitude like the synthetic self-test, so peak
    heights will differ from the ~34 seen there.

    Uses median + MAD (median absolute deviation) rather than
    mean + std for the suggested threshold: matched-filter output
    near a true burst isn't clean white noise (autocorrelation
    sidelobes widen the spread around real peaks), which inflates a
    plain std estimate and can push mean+k*std threshold above the
    real peaks entirely. Median/MAD is robust to that -- it describes
    the bulk noise floor without being dragged up by the (rare) peaks.
    """
    corr_mag = correlate_for_burst(received_iq, reference_iq)
    median = np.median(corr_mag)
    mad = np.median(np.abs(corr_mag - median))
    robust_std = 1.4826 * mad  # scales MAD to be std-equivalent for Gaussian-like data

    print("Correlation magnitude stats (un-thresholded):")
    print(f"  min:            {corr_mag.min():.4f}")
    print(f"  max:            {corr_mag.max():.4f}")
    print(f"  median:         {median:.4f}")
    print(f"  robust std (MAD-based): {robust_std:.4f}")
    print(f"  suggested threshold (median + 8*robust_std): "
          f"{median + 8 * robust_std:.4f}")
    return corr_mag


# ---- Plotting -----------------------------------------------------------
def _get_plt(interactive):
    """Lazily import matplotlib with the right backend. Must happen
    before pyplot is imported anywhere else in the process."""
    import matplotlib
    if not interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_reference_waveform(ref, oversampling=OVERSAMPLING, pilot_bits=PILOT_BITS,
                             plt=None, interactive=True):
    """
    Explain what the pilot waveform itself looks like: I/Q components
    and the instantaneous frequency trajectory (derived from the phase
    derivative), with symbol boundaries and bit values marked. Good for
    a slide introducing "this is what a GFSK pilot burst looks like".
    """
    if plt is None:
        plt = _get_plt(interactive)

    n = len(ref)
    t = np.arange(n)
    # instantaneous frequency = derivative of unwrapped phase
    phase = np.unwrap(np.angle(ref))
    inst_freq = np.diff(phase, prepend=phase[0])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, ref.real, label="I", linewidth=1)
    axes[0].plot(t, ref.imag, label="Q", linewidth=1)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Pilot reference waveform (bits: {pilot_bits})")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, inst_freq, color="tab:red", linewidth=1)
    axes[1].set_ylabel("Instantaneous\nphase step (rad/sample)")
    axes[1].set_xlabel("Sample index")
    axes[1].grid(alpha=0.3)

    # mark symbol boundaries + bit values on both subplots
    for i, bit in enumerate(pilot_bits):
        x = i * oversampling
        for ax in axes:
            ax.axvline(x, color="gray", linestyle=":", alpha=0.4)
        axes[0].text(x + oversampling / 2, axes[0].get_ylim()[1] * 0.85,
                     bit, ha="center", fontsize=9, color="dimgray")

    fig.tight_layout()
    return fig


def plot_correlation_detections(corr_mag, peaks, threshold, sample_rate=None,
                                 plt=None, interactive=True):
    """
    The main "detection" plot for a presentation: correlation magnitude
    across the full capture, threshold line, and each detected burst
    marked and numbered. With ~20 mioty sub-bursts this makes telegram
    splitting visually obvious.
    """
    if plt is None:
        plt = _get_plt(interactive)

    n = len(corr_mag)
    if sample_rate:
        x = np.arange(n) / sample_rate * 1000.0  # milliseconds
        xlabel = "Time (ms)"
    else:
        x = np.arange(n)
        xlabel = "Sample index"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, corr_mag, linewidth=0.7, color="tab:blue", label="Correlation magnitude")
    ax.axhline(threshold, color="tab:red", linestyle="--", linewidth=1,
               label=f"Threshold ({threshold:.3g})")

    for i, (idx, mag) in enumerate(peaks, start=1):
        px = idx / sample_rate * 1000.0 if sample_rate else idx
        ax.plot(px, mag, "o", color="tab:orange", markersize=6, zorder=5)
        ax.annotate(str(i), (px, mag), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="tab:orange")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized correlation magnitude")
    ax.set_title(f"Pilot correlation — {len(peaks)} burst(s) detected")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_burst_alignment(received_iq, reference_iq, sample_idx, plt=None, interactive=True):
    """
    Zoom into one detected burst: overlay the magnitude of the received
    segment against the reference, to visually show why the correlator
    called it a match. Useful as a "here's the evidence" slide.
    """
    if plt is None:
        plt = _get_plt(interactive)

    segment = received_iq[sample_idx: sample_idx + len(reference_iq)]
    t = np.arange(len(reference_iq))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, np.abs(segment), label="Received (|.|)", linewidth=1)
    axes[0].plot(t, np.abs(reference_iq), label="Reference (|.|)", linewidth=1,
                 linestyle="--")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title(f"Burst alignment at sample {sample_idx}")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    # normalized phase trajectories, aligned at start, for shape comparison
    seg_phase = np.unwrap(np.angle(segment)) - np.angle(segment[0])
    ref_phase = np.unwrap(np.angle(reference_iq)) - np.angle(reference_iq[0])
    axes[1].plot(t, seg_phase, label="Received phase", linewidth=1)
    axes[1].plot(t, ref_phase, label="Reference phase", linewidth=1, linestyle="--")
    axes[1].set_ylabel("Unwrapped phase (rad)")
    axes[1].set_xlabel("Sample index (within burst)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def make_presentation_plots(ref, received, corr_mag, peaks, threshold,
                             sample_rate=None, save_dir=None, show=False):
    """
    Generate all three plots and either show them interactively, save
    them as PNGs (for dropping straight into slides), or both.
    """
    import os

    plt = _get_plt(interactive=show)

    fig1 = plot_reference_waveform(ref, plt=plt, interactive=show)
    fig2 = plot_correlation_detections(corr_mag, peaks, threshold,
                                        sample_rate=sample_rate, plt=plt, interactive=show)
    fig3 = None
    if peaks:
        fig3 = plot_burst_alignment(received, ref, peaks[0][0], plt=plt, interactive=show)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig1.savefig(os.path.join(save_dir, "01_reference_waveform.png"),
                     dpi=150, bbox_inches="tight")
        fig2.savefig(os.path.join(save_dir, "02_correlation_detections.png"),
                     dpi=150, bbox_inches="tight")
        if fig3 is not None:
            fig3.savefig(os.path.join(save_dir, "03_burst_alignment.png"),
                         dpi=150, bbox_inches="tight")
        print(f"Saved plots to '{save_dir}/'")

    if show:
        plt.show()


# ---- Demo / self-test ----------------------------------------------------
def _self_test():
    ref = generate_pilot_reference()
    print(f"Pilot reference: {len(ref)} samples "
          f"({len(PILOT_BITS)} symbols x {OVERSAMPLING} oversampling)")

    rng = np.random.default_rng(0)
    noise_len = 5000
    burst_offset = 1234
    noise = (rng.normal(size=noise_len) + 1j * rng.normal(size=noise_len)) * 0.05
    received = noise.astype(np.complex64)
    received[burst_offset:burst_offset + len(ref)] += ref

    detected_start, peak = find_burst_start(received, ref)
    print(f"True offset:     {burst_offset}")
    print(f"Detected offset: {detected_start}")
    print(f"Peak correlation magnitude: {peak:.3f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Correlate a mioty/TS-UNB pilot reference waveform "
                     "against a captured IQ file to detect burst start times."
    )
    parser.add_argument(
        "iq_path", nargs="?", default=None,
        help="Path to the raw IQ capture file (interleaved float32 I/Q, "
             "e.g. MiotyFM_EU1_..._sampleRate228515,616_..._1.iq). "
             "If omitted, runs a synthetic self-test instead.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Minimum normalized correlation magnitude to count as a "
             "detected burst. If omitted, an adaptive threshold "
             "(median + 8*MAD-based robust std of the correlation "
             "output) is computed from your file and used automatically.",
    )
    parser.add_argument(
        "--pilot", type=str, default=PILOT_BITS,
        help=f"Pilot bit sequence (default: {PILOT_BITS}).",
    )
    parser.add_argument(
        "--oversampling", type=int, default=OVERSAMPLING,
        help=f"Samples per symbol (default: {OVERSAMPLING}, matches "
             "the 96x oversampling implied by the filename's "
             "sampleRate = symbol_rate * 96).",
    )
    parser.add_argument(
        "--sample-rate", type=float, default=None,
        help="IQ sample rate in Hz, used to label plot axes in "
             "milliseconds instead of raw sample index. E.g. 228515.616 "
             "for the mioty EU1 UB filename convention.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show plots interactively (reference waveform, correlation "
             "trace with detections, and burst alignment for the first "
             "detected burst).",
    )
    parser.add_argument(
        "--save-plots", type=str, default=None, metavar="DIR",
        help="Save the plots as PNG files into DIR (handy for pasting "
             "straight into slides). Can be combined with --plot.",
    )
    args = parser.parse_args()

    if args.iq_path is None:
        _self_test()
        return

    ref = generate_pilot_reference(args.pilot, oversampling=args.oversampling)
    print(f"Pilot reference: {len(ref)} samples "
          f"({len(args.pilot)} symbols x {args.oversampling} oversampling)")

    received = load_iq_file(args.iq_path)
    print(f"Loaded '{args.iq_path}': {len(received)} IQ samples")

    corr_mag = correlation_diagnostics(received, ref)

    threshold = args.threshold
    if threshold is None:
        median = np.median(corr_mag)
        robust_std = 1.4826 * np.median(np.abs(corr_mag - median))
        threshold = median + 8 * robust_std
        print(f"No --threshold given, using adaptive value: {threshold:.4f}")

    peaks = find_all_bursts(received, ref, threshold=threshold)

    if not peaks:
        print(f"No bursts detected above threshold={threshold:.4f}. "
              "Try lowering --threshold, or check the diagnostics above "
              "for whether the file's dtype/format assumptions are correct.")
        if args.plot or args.save_plots:
            # still worth plotting the reference + correlation trace to
            # diagnose why nothing crossed threshold
            make_presentation_plots(ref, received, corr_mag, peaks, threshold,
                                     sample_rate=args.sample_rate,
                                     save_dir=args.save_plots, show=args.plot)
        return

    print(f"Detected {len(peaks)} burst(s):")
    for idx, (sample_idx, mag) in enumerate(peaks, start=1):
        print(f"  burst {idx:2d}: sample {sample_idx:8d}  "
              f"correlation magnitude {mag:.3f}")

    if args.plot or args.save_plots:
        make_presentation_plots(ref, received, corr_mag, peaks, threshold,
                                 sample_rate=args.sample_rate,
                                 save_dir=args.save_plots, show=args.plot)


if __name__ == "__main__":
    main()