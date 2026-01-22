import pytest
from ase.lattice.cubic import FaceCenteredCubic
from ase.lattice.hexagonal import HexagonalClosedPacked
def test_hcp_miller_lienarly_dependent():
    with pytest.raises((ValueError, NotImplementedError)):
        HexagonalClosedPacked(symbol='Mg', miller=[[1, -1, 0, 0], [1, 0, -1, 0], [0, 1, -1, 0]])