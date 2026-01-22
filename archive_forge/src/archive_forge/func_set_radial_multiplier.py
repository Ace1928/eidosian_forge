import os
import warnings
import time
from typing import Optional
import re
import numpy as np
from ase.units import Hartree
from ase.io.aims import write_aims, read_aims
from ase.data import atomic_numbers
from ase.calculators.calculator import FileIOCalculator, Parameters, kpts2mp, \
def set_radial_multiplier(self):
    assert isinstance(self.radmul, int)
    newctrl = self.ctrlname + '.new'
    fin = open(self.ctrlname, 'r')
    fout = open(newctrl, 'w')
    newline = '    radial_multiplier   %i\n' % self.radmul
    for line in fin:
        if '    radial_multiplier' in line:
            fout.write(newline)
        else:
            fout.write(line)
    fin.close()
    fout.close()
    os.rename(newctrl, self.ctrlname)