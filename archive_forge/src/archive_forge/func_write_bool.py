import os
import numpy as np
from ase.units import Bohr, Ha, Ry, fs, m, s
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx.reader import (read_electron_valency, get_file_name, get_standard_key)
from ase.calculators.openmx import parameters as param
def write_bool(fd, key, value):
    omx_bl = {True: 'On', False: 'Off'}
    fd.write('        '.join([key, '%s' % omx_bl[value]]))
    fd.write('\n')