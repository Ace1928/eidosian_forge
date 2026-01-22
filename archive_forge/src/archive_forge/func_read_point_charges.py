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
def read_point_charges(self):
    """read point charges from previous calculation"""
    pcs = read_data_group('point_charges')
    if len(pcs) > 0:
        lines = pcs.split('\n')[1:]
        charges, positions = ([], [])
        for line in lines:
            columns = [float(col) for col in line.strip().split()]
            positions.append([col * Bohr for col in columns[0:3]])
            charges.append(columns[3])
        self.pcpot = PointChargePotential(charges, positions)