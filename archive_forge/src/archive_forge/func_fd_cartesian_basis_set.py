import copy
from io import StringIO
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.calculator import InputError
from ase.calculators.gaussian import Gaussian
from ase.io import ParseError
from ase.io.gaussian import (_get_atoms_info, _get_cartesian_atom_coords,
@pytest.fixture
def fd_cartesian_basis_set():
    fd_cartesian_basis_set = StringIO('\n    %chk=example.chk\n    %Nprocshared=16\n    %Save\n    # N g1/Gen/TZVPFit ! ASE formatted method and basis\n    # Opt(Tight MaxCyc=100) Integral=Ultrafine\n    Frequency=(ReadIsotopes, Anharmonic)\n\n    Gaussian input prepared by ASE\n\n    0 1\n    O1  -0.464   0.177   0.0\n    H1  -0.464   1.137   0.0\n    H2   0.441  -0.143   0.0\n\n    300 1.0 1.0\n\n    0.1134289259 ! mass of first H\n    ! test comment\n    2 ! mass of 2nd hydrogen\n    ! test comment\n\n\n' + _basis_set_text + '\n')
    return fd_cartesian_basis_set