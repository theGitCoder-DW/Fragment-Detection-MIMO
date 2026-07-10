from midamble import MiotyMidambleGenerator
from visualization import (
    plot_iq,
    plot_phase,
    plot_envelope,
    plot_spectrum
)

generator = MiotyMidambleGenerator()

reference = generator.generate_midamble()

plot_iq(reference, "Reference Midamble")
plot_phase(reference)
plot_envelope(reference)
plot_spectrum(reference, generator.Fs)