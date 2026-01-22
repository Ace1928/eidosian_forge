from math import sqrt
from sys import stdout
import numpy as np
import ase.units as units
from ase.parallel import parprint, paropen
from ase.vibrations import Vibrations
def get_spectrum(self, start=800, end=4000, npts=None, width=4, type='Gaussian', method='standard', direction='central', intensity_unit='(D/A)2/amu', normalize=False):
    """Get infrared spectrum.

        The method returns wavenumbers in cm^-1 with corresponding
        absolute infrared intensity.
        Start and end point, and width of the Gaussian/Lorentzian should
        be given in cm^-1.
        normalize=True ensures the integral over the peaks to give the
        intensity.
        """
    frequencies = self.get_frequencies(method, direction).real
    intensities = self.intensities
    return self.fold(frequencies, intensities, start, end, npts, width, type, normalize)