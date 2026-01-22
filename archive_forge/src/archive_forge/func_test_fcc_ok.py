import pytest
from ase.lattice.cubic import FaceCenteredCubic
from ase.lattice.hexagonal import HexagonalClosedPacked
def test_fcc_ok():
    atoms = FaceCenteredCubic(symbol='Cu', miller=[[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    print(atoms.get_cell())