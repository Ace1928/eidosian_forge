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
def match_kpt_header(line):
    headerpattern = '\\s*kpt#\\s*\\S+\\s*nband=\\s*(\\d+),\\s*wtk=\\s*(\\S+?),\\s*kpt=\\s*(\\S+)+\\s*(\\S+)\\s*(\\S+)'
    m = re.match(headerpattern, line)
    assert m is not None, line
    nbands = int(m.group(1))
    weight = float(m.group(2))
    kvector = np.array(m.group(3, 4, 5)).astype(float)
    return (nbands, weight, kvector)