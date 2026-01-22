import math
import pytest
from chempy import Reaction
from chempy.units import (
from chempy.util._expr import Log10, Constant
from chempy.util.testing import requires
from ..rates import MassAction, Radiolytic
from .._rates import (
@requires('sympy')
def test_PiecewiseTPolyMassAction__sympy():
    import sympy as sp
    tp1 = TPoly([10, 0.1])
    tp2 = ShiftedTPoly([273.15, 37.315, -0.1])
    pwp = MassAction(TPiecewise([0, tp1, 273.15, tp2, 373.15]))
    T = sp.Symbol('T')
    r = Reaction({'A': 2, 'B': 1}, {'C': 1}, inact_reac={'B': 1})
    res1 = pwp({'A': 11, 'B': 13, 'temperature': T}, backend=sp, reaction=r)
    ref1 = 11 ** 2 * 13 * sp.Piecewise((10 + 0.1 * T, sp.And(0 <= T, T <= 273.15)), (37.315 - 0.1 * (T - 273.15), sp.And(273.15 <= T, T <= 373.15)), (sp.Symbol('NAN'), True))
    assert res1 == ref1