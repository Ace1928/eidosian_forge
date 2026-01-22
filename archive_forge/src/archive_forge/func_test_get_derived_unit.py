from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires(units_library)
def test_get_derived_unit():
    registry = SI_base_registry.copy()
    registry['length'] = 0.1 * registry['length']
    conc_unit = get_derived_unit(registry, 'concentration')
    dm = u.decimetre
    assert abs(conc_unit - 1 * u.mole / dm ** 3) < 1e-12 * u.mole / dm ** 3
    registry = defaultdict(lambda: 1)
    registry['amount'] = 1e-09
    assert abs(to_unitless(1.0, get_derived_unit(registry, 'concentration')) - 1000000000.0) < 1e-06