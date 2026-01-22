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
def bandpath2bandpoints(path):
    lines = []
    add = lines.append
    add('BandLinesScale ReciprocalLatticeVectors\n')
    add('%block BandPoints\n')
    for kpt in path.kpts:
        add('    {:18.15f} {:18.15f} {:18.15f}\n'.format(*kpt))
    add('%endblock BandPoints')
    return ''.join(lines)