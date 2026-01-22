import os
from os.path import join
import re
from glob import glob
import warnings
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.units import Hartree, Bohr, fs
from ase.calculators.calculator import Parameters
def write_files_file(fd, label, ppp_list):
    """Write files-file, the file which tells abinit about other files."""
    prefix = label.rsplit('/', 1)[-1]
    fd.write('%s\n' % (prefix + '.in'))
    fd.write('%s\n' % (prefix + '.txt'))
    fd.write('%s\n' % (prefix + 'i'))
    fd.write('%s\n' % (prefix + 'o'))
    fd.write('%s\n' % (prefix + '.abinit'))
    for ppp in ppp_list:
        fd.write('%s\n' % ppp)