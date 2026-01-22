import io
import numpy as np
import warnings
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.io.cif import CIFLoop, parse_loop, NoStructureData, parse_cif
from ase.calculators.calculator import compare_atoms
def test_cif_missingvector(atoms):
    atoms.cell[0] = 0.0
    atoms.pbc[0] = False
    assert atoms.cell.rank == 2
    with pytest.raises(ValueError, match='CIF format can only'):
        roundtrip(atoms)