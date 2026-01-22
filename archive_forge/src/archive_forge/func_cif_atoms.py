import io
import numpy as np
import warnings
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.io.cif import CIFLoop, parse_loop, NoStructureData, parse_cif
from ase.calculators.calculator import compare_atoms
@pytest.fixture
def cif_atoms():
    cif_file = io.StringIO(content)
    return read(cif_file, format='cif')