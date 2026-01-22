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
def read_output(regex):
    """collects all matching strings from the output"""
    hitlist = []
    checkfiles = []
    for filename in os.listdir('.'):
        if filename.startswith('job.') or filename.endswith('.out'):
            checkfiles.append(filename)
    for filename in checkfiles:
        with open(filename, 'rt') as fd:
            lines = fd.readlines()
            for line in lines:
                match = re.search(regex, line)
                if match:
                    hitlist.append(match.group(1))
    return hitlist