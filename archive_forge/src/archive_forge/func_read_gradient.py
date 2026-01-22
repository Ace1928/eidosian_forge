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
def read_gradient(self):
    """read all information in file 'gradient'"""
    from ase import Atom
    grad_string = read_data_group('grad')
    if len(grad_string) == 0:
        return
    lines = grad_string.split('\n')
    history = []
    image = {}
    gradient = []
    atoms = Atoms()
    cycle, energy, norm = (None, None, None)
    for line in lines:
        regex = '^\\s*cycle =\\s*(\\d+)\\s+SCF energy =\\s*([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)\\s+\\|dE\\/dxyz\\| =\\s*([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)'
        match = re.search(regex, line)
        if match:
            if len(atoms):
                image['optimization cycle'] = cycle
                image['total energy'] = energy
                image['gradient norm'] = norm
                image['energy gradient'] = gradient
                history.append(image)
                image = {}
                atoms = Atoms()
                gradient = []
            cycle = int(match.group(1))
            energy = float(match.group(2)) * Ha
            norm = float(match.group(4)) * Ha / Bohr
            continue
        regex = '^\\s*([-+]?[0-9]*\\.?[0-9]+([eEdD][-+]?[0-9]+)?)\\s+([-+]?[0-9]*\\.?[0-9]+([eEdD][-+]?[0-9]+)?)\\s+([-+]?[0-9]*\\.?[0-9]+([eEdD][-+]?[0-9]+)?)\\s+(\\w+)'
        match = re.search(regex, line)
        if match:
            x = float(match.group(1)) * Bohr
            y = float(match.group(3)) * Bohr
            z = float(match.group(5)) * Bohr
            symbol = str(match.group(7)).capitalize()
            if symbol == 'Q':
                symbol = 'X'
            atoms += Atom(symbol, (x, y, z))
            continue
        regex = '^\\s*([-+]?[0-9]*\\.?[0-9]+([eEdD][-+]?[0-9]+)?)\\s+([-+]?[0-9]*\\.?[0-9]+([eEdD][-+]?[0-9]+)?)\\s+([-+]?[0-9]*\\.?[0-9]+([eEdD][-+]?[0-9]+)?)'
        match = re.search(regex, line)
        if match:
            gradx = float(match.group(1).replace('D', 'E')) * Ha / Bohr
            grady = float(match.group(3).replace('D', 'E')) * Ha / Bohr
            gradz = float(match.group(5).replace('D', 'E')) * Ha / Bohr
            gradient.append([gradx, grady, gradz])
    image['optimization cycle'] = cycle
    image['total energy'] = energy
    image['gradient norm'] = norm
    image['energy gradient'] = gradient
    history.append(image)
    self.results['geometry optimization history'] = history