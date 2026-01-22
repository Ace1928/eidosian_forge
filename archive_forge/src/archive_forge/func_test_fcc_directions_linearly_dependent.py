import pytest
from ase.lattice.cubic import FaceCenteredCubic
from ase.lattice.hexagonal import HexagonalClosedPacked
@pytest.mark.parametrize('directions', [[[1, 1, 0], [1, 1, 0], [0, 0, 1]], [[1, 1, 0], [1, 0, 0], [0, 1, 0]]])
def test_fcc_directions_linearly_dependent(directions):
    with pytest.raises(ValueError):
        FaceCenteredCubic(symbol='Cu', directions=directions)