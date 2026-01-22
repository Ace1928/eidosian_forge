from chempy.util.testing import requires
from chempy.units import units_library, default_units as u
from ..numbers import (
def test_number_to_scientific_html():
    assert number_to_scientific_html(2e-17) == '2&sdot;10<sup>-17</sup>'
    assert number_to_scientific_html(1e-17) == '10<sup>-17</sup>'