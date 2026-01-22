from chempy.util.testing import requires
from chempy.units import units_library, default_units as u
from ..numbers import (
@requires(units_library)
def test_number_to_scientific_latex__units():
    assert number_to_scientific_latex(315 * u.km, 17.9 * u.dm, fmt=2) == '315.0000(18)\\,\\mathrm{km}'
    assert number_to_scientific_latex(315 * u.km, 17.9 * u.dm, u.m, fmt=2) == '315000.0(18)\\,\\mathrm{m}'
    assert number_to_scientific_latex(1319 * u.km, 41207 * u.m, u.m, fmt=1) == '1.32(4)\\cdot 10^{6}\\,\\mathrm{m}'