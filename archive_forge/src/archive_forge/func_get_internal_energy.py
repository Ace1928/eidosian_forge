import os
import sys
import numpy as np
from ase import units
def get_internal_energy(self, temperature, verbose=True):
    """Returns the internal energy, in eV, of crystalline solid
        at a specified temperature (K)."""
    self.verbose = verbose
    write = self._vprint
    fmt = '%-15s%13.4f eV'
    if self.formula_units == 0:
        write('Internal energy components at T = %.2f K,\non a per-unit-cell basis:' % temperature)
    else:
        write('Internal energy components at T = %.2f K,\non a per-formula-unit basis:' % temperature)
    write('=' * 31)
    U = 0.0
    omega_e = self.phonon_energies
    dos_e = self.phonon_DOS
    if omega_e[0] == 0.0:
        omega_e = np.delete(omega_e, 0)
        dos_e = np.delete(dos_e, 0)
    write(fmt % ('E_pot', self.potentialenergy))
    U += self.potentialenergy
    zpe_list = omega_e / 2.0
    if self.formula_units == 0:
        zpe = np.trapz(zpe_list * dos_e, omega_e)
    else:
        zpe = np.trapz(zpe_list * dos_e, omega_e) / self.formula_units
    write(fmt % ('E_ZPE', zpe))
    U += zpe
    B = 1.0 / (units.kB * temperature)
    E_vib = omega_e / (np.exp(omega_e * B) - 1.0)
    if self.formula_units == 0:
        E_phonon = np.trapz(E_vib * dos_e, omega_e)
    else:
        E_phonon = np.trapz(E_vib * dos_e, omega_e) / self.formula_units
    write(fmt % ('E_phonon', E_phonon))
    U += E_phonon
    write('-' * 31)
    write(fmt % ('U', U))
    write('=' * 31)
    return U