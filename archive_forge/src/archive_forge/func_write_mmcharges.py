import os
import re
import numpy as np
from ase.units import Hartree, Bohr
from ase.io.orca import write_orca
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
def write_mmcharges(self, filename):
    pc_file = open(os.path.join(self.directory, filename + '.pc'), 'w')
    pc_file.write('{0:d}\n'.format(len(self.mmcharges)))
    for [pos, pc] in zip(self.positions, self.mmcharges):
        [x, y, z] = pos
        pc_file.write('{0:12.6f} {1:12.6f} {2:12.6f} {3:12.6f}\n'.format(pc, x, y, z))
    pc_file.close()