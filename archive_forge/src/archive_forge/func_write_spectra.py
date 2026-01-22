from math import sqrt
from sys import stdout
import numpy as np
import ase.units as units
from ase.parallel import parprint, paropen
from ase.vibrations import Vibrations
def write_spectra(self, out='ir-spectra.dat', start=800, end=4000, npts=None, width=10, type='Gaussian', method='standard', direction='central', intensity_unit='(D/A)2/amu', normalize=False):
    """Write out infrared spectrum to file.

        First column is the wavenumber in cm^-1, the second column the
        absolute infrared intensities, and
        the third column the absorbance scaled so that data runs
        from 1 to 0. Start and end
        point, and width of the Gaussian/Lorentzian should be given
        in cm^-1."""
    energies, spectrum = self.get_spectrum(start, end, npts, width, type, method, direction, normalize)
    spectrum2 = 1.0 - spectrum / spectrum.max()
    outdata = np.empty([len(energies), 3])
    outdata.T[0] = energies
    outdata.T[1] = spectrum
    outdata.T[2] = spectrum2
    with open(out, 'w') as fd:
        fd.write('# %s folded, width=%g cm^-1\n' % (type.title(), width))
        iu, iu_string = self.intensity_prefactor(intensity_unit)
        if normalize:
            iu_string = 'cm ' + iu_string
        fd.write('# [cm^-1] %14s\n' % ('[' + iu_string + ']'))
        for row in outdata:
            fd.write('%.3f  %15.5e  %15.5e \n' % (row[0], iu * row[1], row[2]))