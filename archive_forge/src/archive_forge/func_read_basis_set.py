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
def read_basis_set(self):
    """read the basis set"""
    self.results['basis set'] = []
    self.results['basis set formatted'] = {}
    bsf = read_data_group('basis')
    self.results['basis set formatted']['turbomole'] = bsf
    lines = bsf.split('\n')
    basis_set = {}
    functions = []
    function = {}
    primitives = []
    read_tag = False
    read_data = False
    for line in lines:
        if len(line.strip()) == 0:
            continue
        if '$basis' in line:
            continue
        if '$end' in line:
            break
        if re.match('^\\s*#', line):
            continue
        if re.match('^\\s*\\*', line):
            if read_tag:
                read_tag = False
                read_data = True
            else:
                if read_data:
                    function['primitive functions'] = primitives
                    function['number of primitives'] = len(primitives)
                    primitives = []
                    functions.append(function)
                    function = {}
                    basis_set['functions'] = functions
                    functions = []
                    self.results['basis set'].append(basis_set)
                    basis_set = {}
                    read_data = False
                read_tag = True
            continue
        if read_tag:
            match = re.search('^\\s*(\\w+)\\s+(.+)', line)
            if match:
                basis_set['element'] = match.group(1)
                basis_set['nickname'] = match.group(2)
            else:
                raise RuntimeError('error reading basis set')
        else:
            match = re.search('^\\s+(\\d+)\\s+(\\w+)', line)
            if match:
                if len(primitives) > 0:
                    function['primitive functions'] = primitives
                    function['number of primitives'] = len(primitives)
                    primitives = []
                    functions.append(function)
                    function = {}
                function['shell type'] = str(match.group(2))
                continue
            regex = '^\\s*([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)\\s+([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)'
            match = re.search(regex, line)
            if match:
                exponent = float(match.group(1))
                coefficient = float(match.group(3))
                primitives.append({'exponent': exponent, 'coefficient': coefficient})