import numpy as np
import matplotlib.pyplot as plt


def plot_iq(signal, title="IQ Signal"):

    plt.figure(figsize=(12,4))
    plt.plot(signal.real, label="I")
    plt.plot(signal.imag, label="Q")
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_phase(signal, title="Phase"):

    phase = np.unwrap(np.angle(signal))

    plt.figure(figsize=(12,4))
    plt.plot(phase)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Phase (rad)")
    plt.grid(True)
    plt.show()


def plot_envelope(signal, title="Envelope"):

    envelope = np.abs(signal)

    plt.figure(figsize=(12,4))
    plt.plot(envelope)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()


def plot_spectrum(signal, Fs, title="Spectrum"):

    N = len(signal)

    spectrum = np.fft.fftshift(np.fft.fft(signal))
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=1/Fs))

    plt.figure(figsize=(12,4))
    plt.plot(freq, 20*np.log10(np.abs(spectrum)+1e-12))
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.show()