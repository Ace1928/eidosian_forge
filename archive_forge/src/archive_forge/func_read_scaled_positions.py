import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError
def read_scaled_positions(line, f, debug=None):
    scaled_positions = []
    f.readline()
    f.readline()
    f.readline()
    line = f.readline()
    while not (line == '' or line.isspace()):
        scaled_positions.append(read_tuple_float(line))
        line = f.readline()
    return scaled_positions