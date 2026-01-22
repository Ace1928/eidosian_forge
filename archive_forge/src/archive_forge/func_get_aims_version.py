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
def get_aims_version(string):
    match = re.search('\\s*FHI-aims version\\s*:\\s*(\\S+)', string, re.M)
    return match.group(1)