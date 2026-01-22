from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires(units_library)
def test_unit_registry_from_human_readable():
    hr = unit_registry_to_human_readable(defaultdict(lambda: 1))
    assert hr == dict(((x, (1, 1)) for x in SI_base_registry.keys()))
    ur = unit_registry_from_human_readable(hr)
    assert ur == dict(((x, 1) for x in SI_base_registry.keys()))
    hr = unit_registry_to_human_readable(SI_base_registry)
    assert hr == {'length': (1.0, 'm'), 'mass': (1.0, 'kg'), 'time': (1.0, 's'), 'current': (1.0, 'A'), 'temperature': (1.0, 'K'), 'luminous_intensity': (1.0, 'cd'), 'amount': (1.0, 'mol')}
    ur = unit_registry_from_human_readable(hr)
    assert ur == SI_base_registry
    ur = unit_registry_from_human_readable({'length': (1.0, 'm'), 'mass': (1.0, 'kg'), 'time': (1.0, 's'), 'current': (1.0, 'A'), 'temperature': (1.0, 'K'), 'luminous_intensity': (1.0, 'cd'), 'amount': (1.0, 'mol')})
    assert ur == {'length': u.metre, 'mass': u.kilogram, 'time': u.second, 'current': u.ampere, 'temperature': u.kelvin, 'luminous_intensity': u.candela, 'amount': u.mole}
    ur = unit_registry_from_human_readable({'length': (1000.0, 'm'), 'mass': (0.01, 'kg'), 'time': (10000.0, 's'), 'current': (0.1, 'A'), 'temperature': (10.0, 'K'), 'luminous_intensity': (0.001, 'cd'), 'amount': (10000.0, 'mol')})
    assert ur == {'length': 1000.0 * u.metre, 'mass': 0.01 * u.kilogram, 'time': 10000.0 * u.second, 'current': 0.1 * u.ampere, 'temperature': 10.0 * u.kelvin, 'luminous_intensity': 0.001 * u.candela, 'amount': 10000.0 * u.mole}
    assert ur != {'length': 100.0 * u.metre, 'mass': 0.001 * u.kilogram, 'time': 100.0 * u.second, 'current': 0.01 * u.ampere, 'temperature': 1.0 * u.kelvin, 'luminous_intensity': 0.01 * u.candela, 'amount': 1000.0 * u.mole}