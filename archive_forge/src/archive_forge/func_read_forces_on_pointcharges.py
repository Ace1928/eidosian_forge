import os
import numpy as np
from ase.calculators.calculator import (FileIOCalculator, kpts2ndarray,
from ase.units import Hartree, Bohr
def read_forces_on_pointcharges(self):
    """Read Forces from dftb output file (results.tag)."""
    from ase.units import Hartree, Bohr
    with open(os.path.join(self.directory, 'detailed.out'), 'r') as fd:
        lines = fd.readlines()
    external_forces = []
    for n, line in enumerate(lines):
        if 'Forces on external charges' in line:
            chargestart = n + 1
            break
    else:
        raise RuntimeError('Problem in reading forces on MM external-charges')
    lines1 = lines[chargestart:chargestart + len(self.mmcharges)]
    for line in lines1:
        external_forces.append([float(i) for i in line.split()])
    return np.array(external_forces) * Hartree / Bohr