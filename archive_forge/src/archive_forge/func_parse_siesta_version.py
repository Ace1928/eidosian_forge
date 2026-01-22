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
def parse_siesta_version(output: bytes) -> str:
    match = re.search(b'Siesta Version\\s*:\\s*(\\S+)', output)
    if match is None:
        raise RuntimeError('Could not get Siesta version info from output {!r}'.format(output))
    string = match.group(1).decode('ascii')
    return string