import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import sys


raw = np.fromfile(data_src_IP1, dtype=np.float32)

print(f"Number of float values: {len(raw)}")

iq = raw[0::2] + 1j * raw[1::2]

print(f"Number of IQ samples: {len(iq)}")

print("First 10 IQ samples:")
print(iq[:10])

print(f"Maximum amplitude = {np.max(np.abs(iq)):.3f}")
print(f"Mean amplitude = {np.mean(np.abs(iq)):.3f}")

iq = raw[0::2] + 1j*raw[1::2]

######################################################
# Time domain plot                                 #
######################################################
plt.figure(figsize=(12,4))

plt.plot(np.real(iq[:10000]), label="I")
plt.plot(np.imag(iq[:10000]), label="Q")

plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.title("IQ Samples")

plt.grid(True)
plt.legend()
plt.show()

######################################################
# Constellation plot                                 #
######################################################

plt.figure(figsize=(6,6))

plt.scatter(
    np.real(iq[:50000]),
    np.imag(iq[:50000]),
    s=1
)

plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.title("Constellation")

plt.grid(True)
plt.axis("equal")

plt.show()

######################################################
# Spectrum plot                                      #
######################################################

Fs = 228515.616 # sampling frequency
N = 65536 # nmber of samples of FFT
x = iq[:N]

X = np.fft.fftshift(np.fft.fft(x))

freq = np.fft.fftshift(np.fft.fftfreq(N, d=1/Fs))

plt.figure(figsize=(10,5))
plt.plot(freq, 20*np.log10(np.abs(X)+1e-12))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Spectrum")
plt.grid(True)
plt.show()

######################################################
# Spectogram plot                                    #
######################################################

plt.figure(figsize=(12,6))

plt.specgram(
    iq,
    NFFT=2048,
    Fs=Fs,
    noverlap=1024
)

plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram")

plt.colorbar(label="Power (dB)")

plt.show()