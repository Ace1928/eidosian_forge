from ase.units import Hartree, Bohr
from ase.io import write
import numpy as np
import os
from ase.calculators.calculator import FileIOCalculator
def read_pc_corrections(self):
    """ Crystal calculates Coulomb forces and energies between all
            point charges, and adds that to the QM subsystem. That needs
            to be subtracted again.
            This will be standard in future CRYSTAL versions ."""
    with open(os.path.join(self.directory, 'FORCES_CHG.DAT'), 'r') as infile:
        lines = infile.readlines()
    e = [float(x.split()[-1]) for x in lines if 'SELF-INTERACTION ENERGY(AU)' in x][0]
    e *= Hartree
    f_lines = [s for s in lines if '199' in s]
    assert len(f_lines) == len(self.mmcharges), 'Mismatch in number of point charges from FORCES_CHG.dat'
    pc_forces = np.zeros((len(self.mmcharges), 3))
    for i, l in enumerate(f_lines):
        first = l.split(str(i + 1) + ' 199  ')
        assert len(first) == 2, 'Problem reading FORCES_CHG.dat'
        f = first[-1].split()
        pc_forces[i] = [float(x) for x in f]
    self.coulomb_corrections = (e, pc_forces)