import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError
def read_electron_valency(filename='H_CA13'):
    array = []
    with open(filename, 'r') as fd:
        array = fd.readlines()
        fd.close()
    required_line = ''
    for line in array:
        if 'valence.electron' in line:
            required_line = line
    return rn(required_line)