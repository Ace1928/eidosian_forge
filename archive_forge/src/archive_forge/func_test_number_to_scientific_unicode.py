from chempy.util.testing import requires
from chempy.units import units_library, default_units as u
from ..numbers import (
def test_number_to_scientific_unicode():
    assert number_to_scientific_unicode(2e-17) == u'2·10⁻¹⁷'
    assert number_to_scientific_unicode(1e-17) == u'10⁻¹⁷'