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
def read_vibrational_spectrum(self, noproj=False):
    """Read the vibrational spectrum"""
    self.results['vibrational spectrum'] = []
    key = 'vibrational spectrum'
    if noproj:
        key = 'npr' + key
    lines = read_data_group(key).split('\n')
    for line in lines:
        dictionary = {}
        regex = '^\\s+(\\d+)\\s+(\\S*)\\s+([-+]?\\d+\\.\\d*)\\s+(\\d+\\.\\d*)\\s+(\\S+)\\s+(\\S+)'
        match = re.search(regex, line)
        if match:
            dictionary['mode number'] = int(match.group(1))
            dictionary['irreducible representation'] = str(match.group(2))
            dictionary['frequency'] = {'units': 'cm^-1', 'value': float(match.group(3))}
            dictionary['infrared intensity'] = {'units': 'km/mol', 'value': float(match.group(4))}
            if match.group(5) == 'YES':
                dictionary['infrared active'] = True
            elif match.group(5) == 'NO':
                dictionary['infrared active'] = False
            else:
                dictionary['infrared active'] = None
            if match.group(6) == 'YES':
                dictionary['Raman active'] = True
            elif match.group(6) == 'NO':
                dictionary['Raman active'] = False
            else:
                dictionary['Raman active'] = None
            self.results['vibrational spectrum'].append(dictionary)