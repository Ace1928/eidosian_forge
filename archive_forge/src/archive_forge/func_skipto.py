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
def skipto(string):
    for line in fd:
        if string in line:
            return line
    raise RuntimeError('Not found: {}'.format(string))