import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError
def read_list_bool(line):
    return [read_bool(x) for x in line.split()[1:]]