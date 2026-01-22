import copy
from io import StringIO
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.calculator import InputError
from ase.calculators.gaussian import Gaussian
from ase.io import ParseError
from ase.io.gaussian import (_get_atoms_info, _get_cartesian_atom_coords,
def test_read_gaussian_in_command(fd_command_set):
    with pytest.raises(TypeError):
        read_gaussian_in(fd_command_set, True)