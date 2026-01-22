import os
import re
import warnings
from subprocess import Popen, PIPE
from math import log10, floor
import numpy as np
from ase import Atoms
from ase.units import Ha, Bohr
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import PropertyNotImplementedError, ReadError
def read_data_group(data_group):
    """read a turbomole data group from control file"""
    args = ['sdg', data_group]
    dg = execute(args, error_test=False, stdout_tofile=False)
    return dg.strip()