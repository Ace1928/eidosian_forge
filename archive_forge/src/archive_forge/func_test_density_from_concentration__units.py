from chempy.util.testing import requires
from chempy.units import units_library
@requires(units_library)
def test_density_from_concentration__units():
    from chempy.units import default_units as units
    rho = density_from_concentration(0.4 * units.molar, units=units)
    assert abs(1.024 * units.kg / units.decimetre ** 3 / rho - 1) < 0.001