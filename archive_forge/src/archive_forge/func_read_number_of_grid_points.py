import os
import re
import tempfile
import warnings
import shutil
from os.path import join, isfile, islink
import numpy as np
from ase.units import Ry, eV, Bohr
from ase.data import atomic_numbers
from ase.io.siesta import read_siesta_xv
from ase.calculators.siesta.import_functions import read_rho
from ase.calculators.siesta.import_functions import \
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters, all_changes
from ase.calculators.siesta.parameters import PAOBasisBlock, Species
from ase.calculators.siesta.parameters import format_fdf
def read_number_of_grid_points(self):
    """Read number of grid points from SIESTA's text-output file. """
    fname = self.getpath(ext='out')
    with open(fname, 'r') as fd:
        for line in fd:
            line = line.strip().lower()
            if line.startswith('initmesh: mesh ='):
                n_points = [int(word) for word in line.split()[3:8:2]]
                self.results['n_grid_point'] = n_points
                break
        else:
            raise RuntimeError