from __future__ import annotations
import logging
import math
import os
import subprocess
import tempfile
import time
from shutil import which
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.dev import requires
from monty.json import MSONable, jsanitize
from monty.os import cd
from scipy import constants
from scipy.optimize import fsolve
from scipy.spatial import distance
from pymatgen.core.lattice import Lattice
from pymatgen.core.units import Energy, Length
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine, Kpoint
from pymatgen.electronic_structure.core import Orbital
from pymatgen.electronic_structure.dos import CompleteDos, Dos, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
def read_cube_file(filename):
    """

    Args:
        filename: Cube filename.

    Returns:
        Energy data.
    """
    with open(filename) as file:
        n_atoms = 0
        for idx, line in enumerate(file):
            line = line.rstrip('\n')
            if idx == 0 and 'CUBE' not in line:
                raise ValueError('CUBE file format not recognized')
            if idx == 2:
                tokens = line.split()
                n_atoms = int(tokens[0])
            if idx == 3:
                tokens = line.split()
                n1 = int(tokens[0])
            elif idx == 4:
                tokens = line.split()
                n2 = int(tokens[0])
            elif idx == 5:
                tokens = line.split()
                n3 = int(tokens[0])
            elif idx > 5:
                break
    if 'fort.30' in filename:
        energy_data = np.genfromtxt(filename, skip_header=n_atoms + 6, skip_footer=1)
        n_lines_data = len(energy_data)
        last_line = np.genfromtxt(filename, skip_header=n_lines_data + n_atoms + 6)
        energy_data = np.append(energy_data.flatten(), last_line).reshape(n1, n2, n3)
    elif 'boltztrap_BZ.cube' in filename:
        energy_data = np.loadtxt(filename, skiprows=n_atoms + 6).reshape(n1, n2, n3)
    energy_data /= Energy(1, 'eV').to('Ry')
    return energy_data