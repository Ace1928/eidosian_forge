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
def read_normal_modes(self, noproj=False):
    """Read in vibrational normal modes"""
    self.results['normal modes'] = {}
    self.results['normal modes']['array'] = []
    self.results['normal modes']['projected'] = True
    self.results['normal modes']['mass weighted'] = True
    self.results['normal modes']['units'] = '?'
    dg = read_data_group('nvibro')
    if len(dg) == 0:
        return
    nvibro = int(dg.split()[1])
    self.results['normal modes']['dimension'] = nvibro
    row = []
    key = 'vibrational normal modes'
    if noproj:
        key = 'npr' + key
        self.results['normal modes']['projected'] = False
    lines = read_data_group(key).split('\n')
    for line in lines:
        if key in line:
            continue
        if '$end' in line:
            break
        fields = line.split()
        row.extend(fields[2:len(fields)])
        if len(row) == nvibro:
            float_row = [float(element) for element in row]
            self.results['normal modes']['array'].append(float_row)
            row = []