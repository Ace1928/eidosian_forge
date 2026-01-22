import math
from operator import add
from functools import reduce
import pytest
from chempy import Substance
from chempy.units import (
from ..testing import requires
from ..pyutil import defaultkeydict
from .._expr import (
from ..parsing import parsing_library
@requires(units_library)
def test_Expr_dedimensionalisation__2():
    Poly = Expr.from_callback(_poly, parameter_keys=('E',), argument_names=('x0', Ellipsis))
    T = Poly([3 * u.J, 7 * u.K, 5 * u.K / u.J])
    T = Poly([0.7170172084130019 * u.cal, 12.6 * u.Rankine, 5 * u.K / u.J])
    _ref = 0.8108020083055849
    cv_Al = _get_cv(u.kelvin, u.gram, u.mol)['Al']
    assert isinstance(cv_Al, EinsteinSolid)
    assert cv_Al.args[0] == 0.806 * 428 * u.kelvin
    assert abs(cv_Al({'temperature': 273.15 * u.K, 'molar_gas_constant': 8.3145 * u.J / u.K / u.mol}) - _ref * u.J / u.gram / u.kelvin) < 1e-14
    cv_Al_units, Al_dedim = cv_Al.dedimensionalisation(SI_base_registry)
    assert allclose(cv_Al_units, [u.K, u.kg / u.mol])
    assert isinstance(Al_dedim, EinsteinSolid)
    T_units, dT = T.dedimensionalisation(SI_base_registry)
    assert allclose(T_units, [u.J, u.K, u.K / u.J])
    assert allclose(dT.args, [3, 7, 5])
    assert abs(Al_dedim({'temperature': 273.15, 'molar_gas_constant': 8.3145}) - _ref * 1000) < 1e-14
    assert abs(Al_dedim({'temperature': dT, 'E': (273.15 - 7) / 5 + 3, 'molar_gas_constant': 8.3145}) - _ref * 1000) < 1e-14