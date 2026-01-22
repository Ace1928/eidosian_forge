import os
import sys
import numpy as np
from ase import units
def get_helmholtz_energy(self, temperature, verbose=True):
    """Returns the Helmholtz free energy, in eV, of crystalline solid
        at a specified temperature (K)."""
    self.verbose = True
    write = self._vprint
    U = self.get_internal_energy(temperature, verbose=verbose)
    write('')
    S = self.get_entropy(temperature, verbose=verbose)
    F = U - temperature * S
    write('')
    if self.formula_units == 0:
        write('Helmholtz free energy components at T = %.2f K,\non a per-unit-cell basis:' % temperature)
    else:
        write('Helmholtz free energy components at T = %.2f K,\non a per-formula-unit basis:' % temperature)
    write('=' * 23)
    fmt = '%5s%15.4f eV'
    write(fmt % ('U', U))
    write(fmt % ('-T*S', -temperature * S))
    write('-' * 23)
    write(fmt % ('F', F))
    write('=' * 23)
    return F