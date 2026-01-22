import io
from ase.io import read
from ase.io.wannier90 import read_wout_all
def test_wout_all():
    """Check reading of extra stuff."""
    file = io.StringIO(wout)
    result = read_wout_all(file)
    assert result['spreads'][0] == 0.85842654
    assert abs(result['centers'] - result['atoms'].get_center_of_mass()).max() < 1e-05