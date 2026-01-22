import io
import numpy as np
import warnings
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.io.cif import CIFLoop, parse_loop, NoStructureData, parse_cif
from ase.calculators.calculator import compare_atoms
def test_cif_roundtrip_nonperiodic():
    atoms = molecule('H2O')
    atoms1 = roundtrip(atoms)
    assert not compare_atoms(atoms, atoms1, tol=1e-05)