"""MIOTY/TS-UNB pilot burst detector with 2-D time/frequency correlation.

The pilot is represented using the 4-symbol absolute-phase mapping from the
ETSI MSK/GMSK description. The resulting complex symbol states are converted
to a continuous-phase MSK reference and correlated against the received IQ.

The frequency search produces a time-frequency correlation surface C(tau, f).

Two visualizations are produced in frequency-search mode:

1. Full recording:
   Time vs frequency correlation heatmap.
   Only the strongest candidate peaks are shown.

2. Local correlation surface:
   A high-resolution C(tau, f) surface around the strongest burst.
   It is normalized to 0 dB at the actual maximum and therefore resembles
   the reference ambiguity-function figure.
"""

import numpy as np


# ============================================================================
# MIOTY / MSK CONSTANTS
# ============================================================================

BASE_MULTIPLIER = 15.2587890625

OVERSAMPLING = 96

SYMBOL_RATE_UNITS = BASE_MULTIPLIER * 156
SYMBOL_RATE_HZ = SYMBOL_RATE_UNITS


# ---------------------------------------------------------------------------
# MIOTY pilot
# ---------------------------------------------------------------------------
#
# Raw pilot bits:
#
#       0 1 1 1 0 1 0 0 0 0 1 0
#
# IMPORTANT:
# The bits are NOT directly mapped as:
#
#       0 -> +1
#       1 -> -1
#
# Instead the ETSI MSK/GMSK absolute-phase mapping repeats every 4 symbols.
#
# The resulting mapped pilot is:
#
#       [1, j, 1, -j, 1, j, -1, j, 1, -j, 1, j]
#
# ---------------------------------------------------------------------------

PILOT_BITS = "011101000010"


# Mapping:
#
# symbol position mod 4:
#
# position 0:
#       bit 0 -> +1
#       bit 1 -> -1
#
# position 1:
#       bit 0 -> -j
#       bit 1 -> +j
#
# position 2:
#       bit 0 -> -1
#       bit 1 -> +1
#
# position 3:
#       bit 0 -> +j
#       bit 1 -> -j
#
PHASE_MAP = {
    (0, 0): 1.0 + 0.0j,
    (0, 1): -1.0 + 0.0j,

    (1, 0): -1.0j,
    (1, 1): 1.0j,

    (2, 0): -1.0 + 0.0j,
    (2, 1): 1.0 + 0.0j,

    (3, 0): 1.0j,
    (3, 1): -1.0j,
}


# Explicit mapped pilot for the default pilot bits.
PILOT_MAPPED_SYMBOLS = np.array(
    [
        1,
        1j,
        1,
        -1j,
        1,
        1j,
        -1,
        1j,
        1,
        -1j,
        1,
        1j,
    ],
    dtype=np.complex64,
)

PILOT_PHASES_RAD = np.angle(
    PILOT_MAPPED_SYMBOLS
).astype(np.float64)


# ---------------------------------------------------------------------------
# MIOTY RF constants
# ---------------------------------------------------------------------------

BT = 1.0

# MIOTY EU1 channel spacing equals the symbol rate.
CHANNEL_SPACING_HZ = SYMBOL_RATE_HZ

# EU1 core frame has 24 frequency hops.
NUM_EU1_CHANNELS = 24


# ============================================================================
# PILOT GENERATION
# ============================================================================

def bits_to_array(bit_string):
    """Convert a pilot bit string to an integer NumPy array."""

    if any(b not in "01" for b in bit_string):
        raise ValueError(
            "pilot_bits must contain only 0 and 1"
        )

    return np.array(
        [int(b) for b in bit_string],
        dtype=np.int8,
    )


def map_pilot_bits_to_absolute_phase(
    pilot_bits=PILOT_BITS
):
    """Apply the ETSI 4-symbol absolute-phase mapping."""

    bits = bits_to_array(pilot_bits)

    symbols = np.array(
        [
            PHASE_MAP[
                (i % 4, int(bit))
            ]
            for i, bit in enumerate(bits)
        ],
        dtype=np.complex64,
    )

    return symbols


def generate_cpmsk_pilot_reference(
    pilot_bits=PILOT_BITS,
    oversampling=OVERSAMPLING,
):
    """Generate a continuous-phase MSK reference.

    The ETSI diagram provides the absolute phase state at every symbol
    boundary.

    Adjacent states differ by +/- pi/2.

    We therefore unwrap the phase states and linearly interpolate the
    phase between consecutive symbol boundaries.

    The final symbol is continued using the last observed phase increment.
    """

    symbols = map_pilot_bits_to_absolute_phase(
        pilot_bits
    )

    phases = np.unwrap(
        np.angle(
            symbols.astype(np.complex128)
        )
    )

    if len(phases) == 0:
        return np.empty(
            0,
            dtype=np.complex64
        )

    # Continue the last phase increment for the final symbol.
    if len(phases) > 1:
        last_increment = (
            phases[-1] - phases[-2]
        )
    else:
        last_increment = 0.0

    boundaries = np.concatenate(
        [
            phases,
            [
                phases[-1] + last_increment
            ],
        ]
    )

    n_samples = (
        len(pilot_bits) * oversampling
    )

    phase_samples = np.empty(
        n_samples,
        dtype=np.float64,
    )

    for k in range(len(pilot_bits)):

        phase_start = boundaries[k]
        phase_end = boundaries[k + 1]

        phase_samples[
            k * oversampling:
            (k + 1) * oversampling
        ] = np.linspace(
            phase_start,
            phase_end,
            oversampling,
            endpoint=False,
        )

    return np.exp(
        1j * phase_samples
    ).astype(np.complex64)


def generate_pilot_reference(
    pilot_bits=PILOT_BITS,
    oversampling=OVERSAMPLING,
):
    """Generate the MIOTY pilot reference waveform."""

    return generate_cpmsk_pilot_reference(
        pilot_bits,
        oversampling,
    )


# ============================================================================
# BASIC MATCHED FILTER
# ============================================================================

def correlate_for_burst(
    received_iq,
    reference_iq,
):
    """Calculate normalized pilot correlation."""

    corr = np.correlate(
        received_iq,
        reference_iq,
        mode="valid",
    )

    ref_energy = np.sum(
        np.abs(reference_iq) ** 2
    )

    corr_mag = (
        np.abs(corr)
        / np.sqrt(ref_energy)
    )

    return corr_mag


def find_burst_start(
    received_iq,
    reference_iq,
    threshold=None,
):
    """Find the strongest pilot correlation."""

    corr_mag = correlate_for_burst(
        received_iq,
        reference_iq,
    )

    peak_idx = int(
        np.argmax(corr_mag)
    )

    peak_value = float(
        corr_mag[peak_idx]
    )

    if (
        threshold is not None
        and peak_value < threshold
    ):
        return None, peak_value

    return peak_idx, peak_value


# ============================================================================
# IQ FILE
# ============================================================================

def load_iq_file(
    path,
    dtype=np.complex64,
):
    """Load raw interleaved float32 IQ data."""

    return np.fromfile(
        path,
        dtype=dtype,
    )


# ============================================================================
# FREQUENCY SEARCH GRID
# ============================================================================

def generate_frequency_search_grid(
    num_channels=NUM_EU1_CHANNELS,
    hop_spacing=CHANNEL_SPACING_HZ,
    cfo_span=None,
    freq_step_hz=25.0,
):
    """Generate the fine frequency search grid."""

    if freq_step_hz <= 0:
        raise ValueError(
            "freq_step_hz must be > 0"
        )

    if cfo_span is None:
        cfo_span = hop_spacing

    centers = (
        np.arange(num_channels)
        - (num_channels - 1) / 2.0
    ) * hop_spacing

    half_span = cfo_span / 2.0

    n_side = int(
        np.floor(
            half_span / freq_step_hz
        )
    )

    local = np.arange(
        -n_side * freq_step_hz,
        (n_side + 1) * freq_step_hz,
        freq_step_hz,
        dtype=np.float64,
    )

    freq_offsets = np.concatenate(
        [
            center + local
            for center in centers
        ]
    )

    freq_offsets = np.unique(
        np.round(
            freq_offsets,
            decimals=6,
        )
    )

    return freq_offsets


# ============================================================================
# FFT CORRELATION
# ============================================================================

def correlate_time_frequency_fft(
    received_iq,
    pilot_bits=PILOT_BITS,
    sample_rate=None,
    freq_offsets=None,
    oversampling=OVERSAMPLING,
    batch_size=8,
    threshold=None,
    min_time_spacing=None,
    min_freq_spacing=None,
    max_candidates_per_frequency=16,
    plot_bin_ms=1.0,
    workers=-1,
):
    """FFT-based time-frequency pilot detector.

    The desired quantity is:

        C(tau, f)

    where:

        tau = time/correlation lag
        f   = frequency offset

    This is NOT implemented as a literal np.fft.fft2().

    Instead, FFT-based matched filtering is used independently for each
    frequency hypothesis. This preserves the physical meaning of the
    time axis.
    """

    from scipy.fft import (
        fft,
        ifft,
        next_fast_len,
    )

    from scipy.signal import find_peaks

    if sample_rate is None:
        raise ValueError(
            "sample_rate is required"
        )

    if sample_rate <= 0:
        raise ValueError(
            "sample_rate must be > 0"
        )

    received_iq = np.asarray(
        received_iq,
        dtype=np.complex64,
    )

    reference = generate_pilot_reference(
        pilot_bits,
        oversampling,
    ).astype(np.complex64)

    n_received = len(
        received_iq
    )

    n_reference = len(
        reference
    )

    if n_received < n_reference:
        raise ValueError(
            "Received IQ is shorter than pilot."
        )

    if freq_offsets is None:

        freq_offsets = (
            np.arange(NUM_EU1_CHANNELS)
            - (NUM_EU1_CHANNELS - 1) / 2.0
        ) * CHANNEL_SPACING_HZ

    freq_offsets = np.asarray(
        freq_offsets,
        dtype=np.float64,
    )

    n_time = (
        n_received
        - n_reference
        + 1
    )

    nfft = next_fast_len(
        n_received
        + n_reference
        - 1
    )

    print(
        f"FFT correlation: "
        f"{len(freq_offsets)} frequency hypotheses, "
        f"{n_time} time positions, "
        f"FFT length {nfft}"
    )

    # FFT of received IQ is shared by all frequency hypotheses.
    received_fft = fft(
        received_iq,
        n=nfft,
        workers=workers,
    )

    # ------------------------------------------------------------------------
    # Display surface
    # ------------------------------------------------------------------------

    plot_bin_samples = max(
        1,
        int(
            round(
                sample_rate
                * plot_bin_ms
                / 1000.0
            )
        ),
    )

    n_time_bins = int(
        np.ceil(
            n_time
            / plot_bin_samples
        )
    )

    plot_surface = np.zeros(
        (
            len(freq_offsets),
            n_time_bins,
        ),
        dtype=np.float32,
    )

    candidate_list = []

    noise_samples = []

    ref_energy = np.float32(
        np.sum(
            np.abs(reference) ** 2
        )
    )

    t_reference = (
        np.arange(n_reference)
        / float(sample_rate)
    )

    # ------------------------------------------------------------------------
    # Frequency batches
    # ------------------------------------------------------------------------

    for batch_start in range(
        0,
        len(freq_offsets),
        batch_size,
    ):

        batch_freqs = freq_offsets[
            batch_start:
            batch_start + batch_size
        ]

        batch_length = len(
            batch_freqs
        )

        # ------------------------------------------------------------
        # Frequency-shifted reference bank
        # ------------------------------------------------------------

        references = np.empty(
            (
                batch_length,
                n_reference,
            ),
            dtype=np.complex64,
        )

        for j, frequency in enumerate(
            batch_freqs
        ):

            references[j] = (
                reference
                * np.exp(
                    1j
                    * 2.0
                    * np.pi
                    * frequency
                    * t_reference
                )
            ).astype(
                np.complex64
            )

        # ------------------------------------------------------------
        # FFT matched filtering
        # ------------------------------------------------------------

        matched_filters = np.conj(
            references[:, ::-1]
        )

        ref_fft = fft(
            matched_filters,
            n=nfft,
            axis=1,
            workers=workers,
        )

        corr_full = ifft(
            received_fft[None, :]
            * ref_fft,
            axis=1,
            workers=workers,
        )

        corr = corr_full[
            :,
            n_reference - 1:
            n_received,
        ]

        corr_mag = (
            np.abs(corr)
            / np.sqrt(ref_energy)
        ).astype(
            np.float32
        )

        # ------------------------------------------------------------
        # Build displayed heatmap
        # ------------------------------------------------------------

        for j in range(
            batch_length
        ):

            global_row = (
                batch_start + j
            )

            row = corr_mag[j]

            for time_bin in range(
                n_time_bins
            ):

                start = (
                    time_bin
                    * plot_bin_samples
                )

                end = min(
                    start
                    + plot_bin_samples,
                    n_time,
                )

                plot_surface[
                    global_row,
                    time_bin,
                ] = np.max(
                    row[start:end]
                )

        # ------------------------------------------------------------
        # Threshold estimation
        # ------------------------------------------------------------

        sample_stride = max(
            1,
            n_time // 2000,
        )

        noise_samples.append(
            corr_mag[
                :,
                ::sample_stride,
            ].ravel()
        )

        # ------------------------------------------------------------
        # Candidate local maxima
        # ------------------------------------------------------------

        spacing = (
            min_time_spacing
            if min_time_spacing is not None
            else n_reference
        )

        for j in range(
            batch_length
        ):

            row = corr_mag[j]

            peak_indices, _ = find_peaks(
                row,
                distance=spacing,
            )

            if len(peak_indices) == 0:
                continue

            if (
                len(peak_indices)
                > max_candidates_per_frequency
            ):

                values = row[
                    peak_indices
                ]

                keep = np.argpartition(
                    values,
                    -max_candidates_per_frequency,
                )[
                    -max_candidates_per_frequency:
                ]

                peak_indices = (
                    peak_indices[keep]
                )

            for index in peak_indices:

                candidate_list.append(
                    (
                        int(index),
                        float(batch_freqs[j]),
                        float(row[index]),
                    )
                )

        del (
            references,
            matched_filters,
            ref_fft,
            corr_full,
            corr,
            corr_mag,
        )

    # =========================================================================
    # Threshold
    # =========================================================================

    if threshold is None:

        samples = np.concatenate(
            noise_samples
        )

        median = float(
            np.median(samples)
        )

        mad = float(
            np.median(
                np.abs(
                    samples - median
                )
            )
        )

        robust_std = (
            1.4826 * mad
        )

        threshold = (
            median
            + 8.0 * robust_std
        )

        print(
            "Adaptive threshold: "
            f"{threshold:.4f}"
        )

    else:

        threshold = float(
            threshold
        )

    # =========================================================================
    # 2-D NON-MAXIMUM SUPPRESSION
    # =========================================================================

    if min_time_spacing is None:
        min_time_spacing = n_reference

    if min_freq_spacing is None:

        if len(freq_offsets) > 1:

            grid_step = float(
                np.median(
                    np.diff(
                        freq_offsets
                    )
                )
            )

        else:

            grid_step = 25.0

        # Natural frequency main-lobe width is approximately:
        #
        #       Fs / L
        #
        # where L is the pilot length.
        natural_frequency_width = (
            float(sample_rate)
            / float(n_reference)
        )

        min_freq_spacing = max(
            2.0 * grid_step,
            natural_frequency_width,
        )

    # Strongest first.
    candidate_list.sort(
        key=lambda x: x[2],
        reverse=True,
    )

    detections = []

    for (
        sample_index,
        frequency,
        magnitude,
    ) in candidate_list:

        if magnitude < threshold:
            continue

        too_close = False

        for (
            existing_index,
            existing_frequency,
            _,
        ) in detections:

            if (
                abs(
                    sample_index
                    - existing_index
                )
                < min_time_spacing
                and
                abs(
                    frequency
                    - existing_frequency
                )
                < min_freq_spacing
            ):

                too_close = True
                break

        if not too_close:

            detections.append(
                (
                    sample_index,
                    frequency,
                    magnitude,
                )
            )

    # Sort detections chronologically.
    detections.sort(
        key=lambda x: x[0]
    )

    return (
        freq_offsets,
        plot_surface,
        detections,
        threshold,
    )


# ============================================================================
# FULL-RECORDING PLOT
# ============================================================================

def _get_plt(interactive):
    """Create the correct matplotlib backend."""

    import matplotlib

    if not interactive:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def plot_time_frequency_correlation(
    freq_offsets,
    corr_matrix,
    detections=None,
    global_peak=None,
    sample_rate=None,
    plot_bin_samples=None,
    plt=None,
    interactive=True,
):
    """Plot the full time-frequency correlation map.

    IMPORTANT:
    We deliberately do NOT plot every local correlation peak.

    Only:
        - strongest 24 candidates -> cyan circles
        - global maximum          -> magenta star
    """

    if plt is None:
        plt = _get_plt(
            interactive
        )

    n_channels, n_time_bins = (
        corr_matrix.shape
    )

    if (
        sample_rate is not None
        and plot_bin_samples is not None
    ):

        x = (
            np.arange(n_time_bins)
            * plot_bin_samples
            / sample_rate
            * 1000.0
        )

        xlabel = "Time (ms)"

    else:

        x = np.arange(
            n_time_bins
        )

        xlabel = "Time bin"

    extent = [
        x[0],
        x[-1],
        freq_offsets[0] / 1000.0,
        freq_offsets[-1] / 1000.0,
    ]

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    im = ax.imshow(
        corr_matrix,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
    )

    fig.colorbar(
        im,
        ax=ax,
        label="Correlation magnitude",
    )

    # ------------------------------------------------------------------------
    # ONLY SHOW 24 STRONGEST CANDIDATES
    # ------------------------------------------------------------------------

    if detections:

        selected = sorted(
            detections,
            key=lambda d: d[2],
            reverse=True,
        )[:24]

        for rank, (
            sample_index,
            frequency,
            magnitude,
        ) in enumerate(
            selected,
            start=1,
        ):

            x_position = (
                sample_index
                / sample_rate
                * 1000.0
                if sample_rate
                else sample_index
            )

            ax.scatter(
                x_position,
                frequency / 1000.0,
                s=65,
                facecolors="none",
                edgecolors="cyan",
                linewidths=1.8,
                zorder=6,
            )

            ax.annotate(
                str(rank),
                (
                    x_position,
                    frequency / 1000.0,
                ),
                xytext=(5, 5),
                textcoords="offset points",
                color="cyan",
                fontsize=8,
                fontweight="bold",
                zorder=7,
            )

    # ------------------------------------------------------------------------
    # GLOBAL MAXIMUM
    # ------------------------------------------------------------------------

    if global_peak is not None:

        sample_index, frequency, magnitude = (
            global_peak
        )

        x_position = (
            sample_index
            / sample_rate
            * 1000.0
            if sample_rate
            else sample_index
        )

        ax.scatter(
            x_position,
            frequency / 1000.0,
            s=180,
            marker="*",
            facecolors="magenta",
            edgecolors="black",
            linewidths=1.4,
            zorder=10,
            label="Global maximum",
        )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        "Frequency offset (kHz)"
    )

    ax.set_title(
        "MIOTY pilot correlation — time-frequency search"
    )

    ax.grid(
        alpha=0.25
    )

    if global_peak is not None:
        ax.legend(
            loc="upper right"
        )

    fig.tight_layout()

    return fig


# ============================================================================
# LOCAL 2-D CORRELATION SURFACE
# ============================================================================

def compute_local_correlation_surface(
    received_iq,
    reference_iq,
    sample_rate,
    center_sample,
    center_freq_hz,
    time_half_width_ms=10.0,
    freq_half_width_hz=1000.0,
    freq_step_hz=25.0,
):
    """Compute a high-resolution local C(tau, f) surface.

    This is the important plot corresponding to the reference figure.

    The correlation is evaluated around the strongest burst.

    The resulting surface is normalized so that:

        maximum = 0 dB

    and the axes are shifted so that the actual discrete maximum occurs at:

        time offset    = 0 ms
        frequency      = 0 Hz
    """

    from scipy.signal import correlate

    # ------------------------------------------------------------------------
    # Extract a local received-signal region.
    # ------------------------------------------------------------------------

    half_samples = max(
        1,
        int(
            round(
                time_half_width_ms
                * sample_rate
                / 1000.0
            )
        ),
    )

    start = max(
        0,
        int(center_sample)
        - half_samples,
    )

    end = min(
        len(received_iq),
        int(center_sample)
        + half_samples
        + len(reference_iq),
    )

    segment = np.asarray(
        received_iq[start:end],
        dtype=np.complex64,
    )

    # ------------------------------------------------------------------------
    # Frequency grid around the detected maximum.
    # ------------------------------------------------------------------------

    freq_offsets = np.arange(
        center_freq_hz
        - freq_half_width_hz,

        center_freq_hz
        + freq_half_width_hz
        + 0.5 * freq_step_hz,

        freq_step_hz,

        dtype=np.float64,
    )

    n_time = (
        len(segment)
        - len(reference_iq)
        + 1
    )

    if n_time <= 0:

        raise ValueError(
            "Local segment is shorter than reference waveform."
        )

    reference_time = (
        np.arange(
            len(reference_iq)
        )
        / sample_rate
    )

    surface = np.empty(
        (
            len(freq_offsets),
            n_time,
        ),
        dtype=np.float64,
    )

    reference_energy = np.sum(
        np.abs(reference_iq) ** 2
    )

    # ------------------------------------------------------------------------
    # Calculate C(tau, f)
    # ------------------------------------------------------------------------

    for row, frequency in enumerate(
        freq_offsets
    ):

        shifted_reference = (
            reference_iq
            * np.exp(
                1j
                * 2.0
                * np.pi
                * frequency
                * reference_time
            )
        )

        correlation = correlate(
            segment,
            shifted_reference,
            mode="valid",
            method="fft",
        )

        surface[row] = (
            np.abs(correlation)
            / np.sqrt(
                reference_energy
            )
        )

    # ------------------------------------------------------------------------
    # Find actual 2-D maximum.
    # ------------------------------------------------------------------------

    max_row, max_column = (
        np.unravel_index(
            np.argmax(surface),
            surface.shape,
        )
    )

    maximum = surface[
        max_row,
        max_column,
    ]

    # Absolute location of maximum.
    peak_sample = (
        start
        + max_column
    )

    peak_frequency = (
        freq_offsets[max_row]
    )

    # ------------------------------------------------------------------------
    # Re-center axes around the actual maximum.
    # ------------------------------------------------------------------------

    time_offsets_ms = (
        (
            np.arange(n_time)
            - max_column
        )
        / sample_rate
        * 1000.0
    )

    frequency_offsets_hz = (
        freq_offsets
        - peak_frequency
    )

    # ------------------------------------------------------------------------
    # Normalize to 0 dB.
    # ------------------------------------------------------------------------

    surface_db = (
        20.0
        * np.log10(
            np.maximum(
                surface / maximum,
                1e-12,
            )
        )
    )

    # Limit display to -21 dB.
    surface_db = np.maximum(
        surface_db,
        -21.0,
    )

    refined_peak = (
        int(peak_sample),
        float(peak_frequency),
        float(maximum),
    )

    return (
        time_offsets_ms,
        frequency_offsets_hz,
        surface_db,
        refined_peak,
    )


def plot_local_correlation_surface(
    time_offsets_ms,
    freq_offsets_hz,
    surface_db,
    peak_info=None,
    plt=None,
    interactive=True,
):
    """Plot the local correlation surface like the reference figure."""

    if plt is None:
        plt = _get_plt(
            interactive
        )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    # Exactly the contour levels used conceptually in the reference:
    #
    #     0, -3, -6, ... -21 dB
    #
    levels = np.arange(
        -21,
        0.01,
        3,
    )

    contour = ax.contourf(
        time_offsets_ms,
        freq_offsets_hz / 1000.0,
        surface_db,
        levels=levels,
        cmap="viridis",
        extend="min",
    )

    fig.colorbar(
        contour,
        ax=ax,
        ticks=[
            -21,
            -18,
            -15,
            -12,
            -9,
            -6,
            -3,
            0,
        ],
        label="Normalized correlation [dB]",
    )

    # ------------------------------------------------------------------------
    # Exact maximum
    #
    # The maximum is at (0 ms, 0 Hz).
    #
    # Use a small black point so we don't hide the yellow region.
    # ------------------------------------------------------------------------

    ax.scatter(
        0.0,
        0.0,
        s=35,
        c="black",
        marker="o",
        zorder=10,
    )

    # Crosshair.
    ax.axvline(
        0.0,
        color="white",
        linewidth=0.8,
        alpha=0.5,
    )

    ax.axhline(
        0.0,
        color="white",
        linewidth=0.8,
        alpha=0.5,
    )

    ax.set_xlabel(
        "Time Offset [ms]"
    )

    ax.set_ylabel(
        "Frequency Offset [kHz]"
    )

    ax.set_title(
        "2-D local pilot correlation around strongest burst"
    )

    ax.set_xlim(
        time_offsets_ms[0],
        time_offsets_ms[-1],
    )

    ax.set_ylim(
        freq_offsets_hz[0] / 1000.0,
        freq_offsets_hz[-1] / 1000.0,
    )

    ax.grid(
        alpha=0.25,
        color="white",
    )

    fig.tight_layout()

    return fig


# ============================================================================
# SELF TEST
# ============================================================================

def _self_test():

    reference = generate_pilot_reference()

    print(
        f"Pilot reference: "
        f"{len(reference)} samples "
        f"({len(PILOT_BITS)} symbols x "
        f"{OVERSAMPLING} oversampling)"
    )

    print(
        "Pilot bits:",
        PILOT_BITS,
    )

    print(
        "Mapped pilot:",
        map_pilot_bits_to_absolute_phase(
            PILOT_BITS
        ),
    )

    rng = np.random.default_rng(
        0
    )

    noise_length = 5000

    burst_offset = 1234

    noise = (
        rng.normal(
            size=noise_length
        )
        +
        1j
        * rng.normal(
            size=noise_length
        )
    ) * 0.05

    received = noise.astype(
        np.complex64
    )

    received[
        burst_offset:
        burst_offset
        + len(reference)
    ] += reference

    detected_start, peak = (
        find_burst_start(
            received,
            reference,
        )
    )

    print(
        f"True offset:     "
        f"{burst_offset}"
    )

    print(
        f"Detected offset: "
        f"{detected_start}"
    )

    print(
        f"Peak correlation: "
        f"{peak:.3f}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "MIOTY/TS-UNB pilot burst detector "
            "with 2-D time/frequency correlation."
        )
    )

    parser.add_argument(
        "iq_path",
        nargs="?",
        default=None,
        help=(
            "Raw IQ file. If omitted, "
            "runs synthetic self-test."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Correlation detection threshold. "
            "If omitted, adaptive threshold is used."
        ),
    )

    parser.add_argument(
        "--pilot",
        type=str,
        default=PILOT_BITS,
        help=(
            f"Pilot bits "
            f"(default: {PILOT_BITS})."
        ),
    )

    parser.add_argument(
        "--oversampling",
        type=int,
        default=OVERSAMPLING,
        help=(
            f"Samples per symbol "
            f"(default: {OVERSAMPLING})."
        ),
    )

    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help=(
            "IQ sample rate in Hz. "
            "For your capture: 228515.616 Hz."
        ),
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show plots interactively.",
    )

    parser.add_argument(
        "--save-plots",
        type=str,
        default=None,
        metavar="DIR",
        help="Save generated plots to this directory.",
    )

    parser.add_argument(
        "--freq-search",
        action="store_true",
        help=(
            "Run the 2-D time-frequency "
            "pilot detector."
        ),
    )

    parser.add_argument(
        "--num-channels",
        type=int,
        default=NUM_EU1_CHANNELS,
        help=(
            f"Number of MIOTY EU1 "
            f"frequency channels "
            f"(default: {NUM_EU1_CHANNELS})."
        ),
    )

    parser.add_argument(
        "--freq-step",
        type=float,
        default=25.0,
        help=(
            "Frequency-search step in Hz "
            "(default: 25 Hz)."
        ),
    )

    parser.add_argument(
        "--cfo-span",
        type=float,
        default=None,
        help=(
            "Frequency span around each "
            "MIOTY channel."
        ),
    )

    parser.add_argument(
        "--freq-batch-size",
        type=int,
        default=8,
        help=(
            "Number of frequency hypotheses "
            "processed per batch."
        ),
    )

    parser.add_argument(
        "--plot-bin-ms",
        type=float,
        default=1.0,
        help=(
            "Time bin used for full-recording "
            "heatmap display."
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------------
    # No IQ file -> self test
    # ------------------------------------------------------------------------

    if args.iq_path is None:

        _self_test()

        return

    # ------------------------------------------------------------------------
    # Generate pilot
    # ------------------------------------------------------------------------

    reference = generate_pilot_reference(
        args.pilot,
        args.oversampling,
    )

    mapped_pilot = (
        map_pilot_bits_to_absolute_phase(
            args.pilot
        )
    )

    print(
        f"Pilot reference: "
        f"{len(reference)} samples "
        f"({len(args.pilot)} symbols x "
        f"{args.oversampling} oversampling)"
    )

    print(
        "Mapped pilot symbols:",
        mapped_pilot.tolist(),
    )

    # ------------------------------------------------------------------------
    # Load IQ
    # ------------------------------------------------------------------------

    received = load_iq_file(
        args.iq_path
    )

    print(
        f"Loaded '{args.iq_path}': "
        f"{len(received)} IQ samples"
    )

    # =========================================================================
    # 2-D FREQUENCY SEARCH
    # =========================================================================

    if args.freq_search:

        if args.sample_rate is None:

            print(
                "--freq-search requires "
                "--sample-rate."
            )

            return

        # ------------------------------------------------------------
        # Frequency grid
        # ------------------------------------------------------------

        freq_offsets = (
            generate_frequency_search_grid(
                num_channels=args.num_channels,
                hop_spacing=CHANNEL_SPACING_HZ,
                cfo_span=args.cfo_span,
                freq_step_hz=args.freq_step,
            )
        )

        print(
            f"Searching "
            f"{len(freq_offsets)} "
            f"frequency hypotheses."
        )

        print(
            f"Frequency range: "
            f"{freq_offsets[0]:+.1f} Hz "
            f"... "
            f"{freq_offsets[-1]:+.1f} Hz"
        )

        # ------------------------------------------------------------
        # Run 2-D correlation
        # ------------------------------------------------------------

        (
            freq_offsets,
            corr_surface,
            detections,
            threshold,
        ) = correlate_time_frequency_fft(
            received,
            pilot_bits=args.pilot,
            sample_rate=args.sample_rate,
            freq_offsets=freq_offsets,
            oversampling=args.oversampling,
            batch_size=args.freq_batch_size,
            threshold=args.threshold,
            plot_bin_ms=args.plot_bin_ms,
        )

        # ------------------------------------------------------------
        # Print detections
        # ------------------------------------------------------------

        if not detections:

            print(
                f"No bursts detected above "
                f"threshold={threshold:.4f}."
            )

        else:

            print(
                f"Detected "
                f"{len(detections)} "
                f"candidate bursts."
            )

            for i, (
                sample_index,
                frequency,
                magnitude,
            ) in enumerate(
                detections,
                start=1,
            ):

                time_ms = (
                    sample_index
                    / args.sample_rate
                    * 1000.0
                )

                print(
                    f"  burst {i:3d}: "
                    f"sample={sample_index:8d} "
                    f"time={time_ms:10.3f} ms "
                    f"freq={frequency:+10.2f} Hz "
                    f"corr={magnitude:.4f}"
                )

        # ============================================================
        # PLOTS
        # ============================================================

        if (
            args.plot
            or args.save_plots
        ):

            plt = _get_plt(
                interactive=args.plot
            )

            plot_bin_samples = max(
                1,
                int(
                    round(
                        args.sample_rate
                        * args.plot_bin_ms
                        / 1000.0
                    )
                ),
            )

            # --------------------------------------------------------
            # Global maximum
            #
            # IMPORTANT:
            # This is the strongest correlation among the actual
            # full-resolution detections.
            # --------------------------------------------------------

            global_peak = (
                max(
                    detections,
                    key=lambda d: d[2],
                )
                if detections
                else None
            )

            # --------------------------------------------------------
            # Full recording plot
            # --------------------------------------------------------

            fig = (
                plot_time_frequency_correlation(
                    freq_offsets,
                    corr_surface,
                    detections=detections,
                    global_peak=global_peak,
                    sample_rate=args.sample_rate,
                    plot_bin_samples=plot_bin_samples,
                    plt=plt,
                    interactive=args.plot,
                )
            )

            # --------------------------------------------------------
            # Local 2-D correlation surface
            # --------------------------------------------------------

            fig_local = None

            if global_peak is not None:

                (
                    tau_ms,
                    delta_f_hz,
                    local_db,
                    refined_peak,
                ) = (
                    compute_local_correlation_surface(
                        received_iq=received,
                        reference_iq=reference,
                        sample_rate=args.sample_rate,
                        center_sample=global_peak[0],
                        center_freq_hz=global_peak[1],

                        # Reference-style local window
                        time_half_width_ms=10.0,
                        freq_half_width_hz=1000.0,

                        # Your fine frequency search
                        freq_step_hz=args.freq_step,
                    )
                )

                fig_local = (
                    plot_local_correlation_surface(
                        tau_ms,
                        delta_f_hz,
                        local_db,
                        peak_info=refined_peak,
                        plt=plt,
                        interactive=args.plot,
                    )
                )

                print(
                    "\nStrongest correlation:"
                )

                print(
                    f"  sample:    "
                    f"{refined_peak[0]}"
                )

                print(
                    f"  time:      "
                    f"{refined_peak[0] / args.sample_rate * 1000:.3f} ms"
                )

                print(
                    f"  frequency: "
                    f"{refined_peak[1]:+.2f} Hz"
                )

                print(
                    f"  magnitude: "
                    f"{refined_peak[2]:.4f}"
                )

            # --------------------------------------------------------
            # Save plots
            # --------------------------------------------------------

            if args.save_plots:

                import os

                os.makedirs(
                    args.save_plots,
                    exist_ok=True,
                )

                full_path = os.path.join(
                    args.save_plots,
                    "04_time_frequency_2d_fft_correlation.png",
                )

                fig.savefig(
                    full_path,
                    dpi=150,
                    bbox_inches="tight",
                )

                print(
                    f"Saved '{full_path}'"
                )

                if fig_local is not None:

                    local_path = os.path.join(
                        args.save_plots,
                        "05_local_global_max_correlation.png",
                    )

                    fig_local.savefig(
                        local_path,
                        dpi=150,
                        bbox_inches="tight",
                    )

                    print(
                        f"Saved '{local_path}'"
                    )

            if args.plot:

                plt.show()

        return

    # =========================================================================
    # SINGLE-FREQUENCY DETECTION
    # =========================================================================

    corr_mag = correlate_for_burst(
        received,
        reference,
    )

    # Adaptive threshold.
    threshold = args.threshold

    if threshold is None:

        median = np.median(
            corr_mag
        )

        robust_std = (
            1.4826
            * np.median(
                np.abs(
                    corr_mag
                    - median
                )
            )
        )

        threshold = (
            median
            + 8
            * robust_std
        )

    print(
        f"Detection threshold: "
        f"{threshold:.4f}"
    )

    from scipy.signal import find_peaks

    peaks_idx, _ = find_peaks(
        corr_mag,
        height=threshold,
        distance=len(reference),
    )

    peaks = [
        (
            int(index),
            float(corr_mag[index]),
        )
        for index in peaks_idx
    ]

    if not peaks:

        print(
            "No bursts detected."
        )

        return

    print(
        f"Detected {len(peaks)} burst(s)."
    )

    for i, (
        index,
        magnitude,
    ) in enumerate(
        peaks,
        start=1,
    ):

        print(
            f"  burst {i:3d}: "
            f"sample={index:8d} "
            f"correlation={magnitude:.4f}"
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()