from collections import defaultdict
import pytest
from ..util.testing import requires
from ..units import (
@requires(units_library)
def test_to_unitless():
    dm = u.decimetre
    vals = [1.0 * dm, 2.0 * dm]
    result = to_unitless(vals, u.metre)
    assert result[0] == 0.1
    assert result[1] == 0.2
    with pytest.raises(ValueError):
        to_unitless([42, 43], u.metre)
    with pytest.raises(ValueError):
        to_unitless(np.array([42, 43]), u.metre)
    vals = [1.0, 2.0] * dm
    result = to_unitless(vals, u.metre)
    assert result[0] == 0.1
    assert result[1] == 0.2
    length_unit = 1000 * u.metre
    result = to_unitless(1.0 * u.metre, length_unit)
    assert abs(result - 0.001) < 1e-12
    amount_unit = 1e-09
    assert abs(to_unitless(1.0, amount_unit) - 1000000000.0) < 1e-06
    assert abs(to_unitless(3 / (u.second * u.molar), u.metre ** 3 / u.mole / u.second) - 0.003) < 1e-12
    assert abs(to_unitless(2 * u.dm3, u.cm3) - 2000) < 1e-12
    assert abs(to_unitless(2 * u.m3, u.dm3) - 2000) < 1e-12
    assert float(to_unitless(UncertainQuantity(2, u.dm3, 0.3), u.cm3)) - 2000 < 1e-12
    g1 = UncertainQuantity(4.46, u.per100eV, 0)
    g_unit = get_derived_unit(SI_base_registry, 'radiolytic_yield')
    assert abs(to_unitless(g1, g_unit) - 4.46 * 1.036e-07) < 1e-09
    g2 = UncertainQuantity(-4.46, u.per100eV, 0)
    assert abs(to_unitless(-g2, g_unit) - 4.46 * 1.036e-07) < 1e-09
    vals = np.array([1.0 * dm, 2.0 * dm], dtype=object)
    result = to_unitless(vals, u.metre)
    assert result[0] == 0.1
    assert result[1] == 0.2
    one_billionth_molar_in_nanomolar = to_unitless(1e-09 * u.molar, u.nanomolar)
    assert one_billionth_molar_in_nanomolar == 1