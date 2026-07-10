"""
midamble.py

Generate the MIOTY reference midamble waveform.

Current implementation:
    - NRZ mapping
    - Continuous Phase MSK
    - 96 samples/symbol

Future improvements:
    - Gaussian pulse shaping (from Fraunhofer transmitter)
"""

import numpy as np


class MiotyMidambleGenerator:

    def __init__(
        self,
        sample_rate=228515.616,
        symbol_rate=2380.371,
        modulation_index=0.5
    ):

        self.Fs = sample_rate
        self.Rs = symbol_rate
        self.h = modulation_index

        self.sps = int(round(self.Fs / self.Rs))

    ######################################################################
    # Bit Mapping
    ######################################################################

    @staticmethod
    def bits_to_symbols(bits):
        """
        Convert bits to NRZ symbols.

        0 -> -1
        1 -> +1
        """

        bits = np.asarray(bits)

        return 2 * bits - 1

    ######################################################################
    # MSK Modulator
    ######################################################################

    def modulate(self, bits):

        symbols = self.bits_to_symbols(bits)

        waveform = []

        phase = 0.0

        # Phase increment per sample
        dphi = np.pi * self.h / self.sps

        for symbol in symbols:

            for _ in range(self.sps):

                phase += symbol * dphi

                waveform.append(
                    np.exp(1j * phase)
                )

        return np.asarray(waveform, dtype=np.complex64)

    ######################################################################
    # Midamble
    ######################################################################

    def generate_midamble(self):

        pilot_bits = np.array(
            [
                0,1,1,1,
                0,1,0,0,
                0,0,1,0
            ],
            dtype=np.uint8
        )

        return self.modulate(pilot_bits)
    