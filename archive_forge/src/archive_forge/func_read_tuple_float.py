import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError
def read_tuple_float(line):
    return tuple([float(x) for x in line.split()[-3:]])