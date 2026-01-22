import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError
def read_stress_tensor(line, f, debug=None):
    f.readline()
    f.readline()
    line = f.readline()
    xx, xy, xz = read_tuple_float(line)
    line = f.readline()
    yx, yy, yz = read_tuple_float(line)
    line = f.readline()
    zx, zy, zz = read_tuple_float(line)
    stress = [xx, yy, zz, (zy + yz) / 2, (zx + xz) / 2, (yx + xy) / 2]
    return stress