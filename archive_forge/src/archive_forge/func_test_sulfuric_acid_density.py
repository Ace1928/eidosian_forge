from chempy.util.testing import requires
from chempy.units import units_library
@requires('numpy')
def test_sulfuric_acid_density():
    rho = sulfuric_acid_density(0.1, 298)
    assert abs(1063.8 - rho) < 0.1