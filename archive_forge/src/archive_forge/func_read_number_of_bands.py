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
def read_number_of_bands(self):
    nband = None
    lines = open(self.out, 'r').readlines()
    for n, line in enumerate(lines):
        if line.rfind('Number of Kohn-Sham states') > -1:
            nband = int(line.split(':')[-1].strip())
    return nband