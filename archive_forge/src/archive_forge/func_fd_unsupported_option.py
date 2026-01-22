import copy
from io import StringIO
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.calculator import InputError
from ase.calculators.gaussian import Gaussian
from ase.io import ParseError
from ase.io.gaussian import (_get_atoms_info, _get_cartesian_atom_coords,
def fd_unsupported_option():
    unsupported_text = ''
    for i, line in enumerate(_zmatrix_file_text.split('\n')):
        if i == 4:
            unsupported_text += 'Geom=ModRedundant freq=ReadIso\n'
        else:
            unsupported_text += line + '\n'
    return StringIO(unsupported_text)