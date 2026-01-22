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
def read_datetime(self):
    """read the datetime of the most recent calculation
        from the tm output if stored in a file
        """
    datetimes = read_output('(\\d{4}-[01]\\d-[0-3]\\d([T\\s][0-2]\\d:[0-5]\\d:[0-5]\\d\\.\\d+)?([+-][0-2]\\d:[0-5]\\d|Z)?)')
    if len(datetimes) == 0:
        warnings.warn('no turbomole datetime detected')
        self.datetime = None
    else:
        self.datetime = sorted(datetimes, reverse=True)[0]