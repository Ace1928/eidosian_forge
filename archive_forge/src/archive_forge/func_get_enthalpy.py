import os
import sys
import numpy as np
from ase import units
def get_enthalpy(self, temperature, verbose=True):
    """Returns the enthalpy, in eV, in the ideal gas approximation
        at a specified temperature (K)."""
    self.verbose = verbose
    write = self._vprint
    fmt = '%-15s%13.3f eV'
    write('Enthalpy components at T = %.2f K:' % temperature)
    write('=' * 31)
    H = 0.0
    write(fmt % ('E_pot', self.potentialenergy))
    H += self.potentialenergy
    zpe = self.get_ZPE_correction()
    write(fmt % ('E_ZPE', zpe))
    H += zpe
    Cv_t = 3.0 / 2.0 * units.kB
    write(fmt % ('Cv_trans (0->T)', Cv_t * temperature))
    H += Cv_t * temperature
    if self.geometry == 'nonlinear':
        Cv_r = 3.0 / 2.0 * units.kB
    elif self.geometry == 'linear':
        Cv_r = units.kB
    elif self.geometry == 'monatomic':
        Cv_r = 0.0
    write(fmt % ('Cv_rot (0->T)', Cv_r * temperature))
    H += Cv_r * temperature
    dH_v = self._vibrational_energy_contribution(temperature)
    write(fmt % ('Cv_vib (0->T)', dH_v))
    H += dH_v
    Cp_corr = units.kB * temperature
    write(fmt % ('(C_v -> C_p)', Cp_corr))
    H += Cp_corr
    write('-' * 31)
    write(fmt % ('H', H))
    write('=' * 31)
    return H