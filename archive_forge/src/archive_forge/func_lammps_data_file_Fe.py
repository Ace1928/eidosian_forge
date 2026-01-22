import pytest
import numpy as np
from ase import Atom
from ase.build import bulk
import ase.io
from ase import units
from ase.md.verlet import VelocityVerlet
@pytest.fixture
def lammps_data_file_Fe(datadir):
    return datadir / 'lammpslib_simple_input.data'