import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError
def read_energies(line, f, debug=None):
    line = f.readline()
    if '***' in line:
        point = 7
    else:
        point = 16
    for i in range(point):
        f.readline()
    line = f.readline()
    energies = []
    while not (line == '' or line.isspace()):
        energies.append(float(line.split()[2]))
        line = f.readline()
    return energies