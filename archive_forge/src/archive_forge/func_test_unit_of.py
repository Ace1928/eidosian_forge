from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires(units_library)
def test_unit_of():
    assert compare_equality(unit_of(0.1 * u.metre / u.second), u.metre / u.second)
    assert not compare_equality(unit_of(0.1 * u.metre / u.second), u.kilometre / u.second)
    assert compare_equality(unit_of(7), 1)
    assert unit_of(u.gray).dimensionality == u.gray.dimensionality
    ref = (u.joule / u.kg).simplified.dimensionality
    assert unit_of(u.gray, simplified=True).dimensionality == ref
    assert compare_equality(unit_of(dict(foo=3 * u.molar, bar=2 * u.molar)), u.molar)
    assert not compare_equality(unit_of(dict(foo=3 * u.molar, bar=2 * u.molar)), u.second)
    with pytest.raises(Exception):
        unit_of(dict(foo=3 * u.molar, bar=2 * u.second))
    assert not compare_equality(unit_of(dict(foo=3 * u.molar, bar=2 * u.molar)), u.mol / u.metre ** 3)