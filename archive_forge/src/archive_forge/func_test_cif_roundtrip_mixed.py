import io
import numpy as np
import warnings
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.io.cif import CIFLoop, parse_loop, NoStructureData, parse_cif
from ase.calculators.calculator import compare_atoms
def test_cif_roundtrip_mixed():
    atoms = Atoms('Au', cell=[1.0, 2.0, 3.0], pbc=[1, 1, 0])
    atoms1 = roundtrip(atoms)
    assert all(atoms1.pbc)
    assert compare_atoms(atoms, atoms1, tol=1e-05) == ['pbc']
    assert atoms.get_scaled_positions() == pytest.approx(atoms1.get_scaled_positions(), abs=1e-05)