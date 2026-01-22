import math
import pytest
from chempy import Reaction
from chempy.units import (
from chempy.util._expr import Log10, Constant
from chempy.util.testing import requires
from ..rates import MassAction, Radiolytic
from .._rates import (
def test_Log10PiecewiseRTPolyMassAction():
    p1 = RTPoly([12.281, -376.8, -66730.0, -10750000.0])
    p2 = RTPoly([-47.532, 4.92, -1.036, 0.0])
    ratex = MassAction(10 ** TPiecewise([293.15, p1, 423.15, p2, 623.15]))
    r = Reaction({'e-(aq)': 2}, {'H2': 1, 'OH-': 2}, ratex, {'H2O': 2})
    res = ratex({'e-(aq)': 1e-13, 'temperature': 293.15}, reaction=r)
    ref = 6200000000.0 * 1e-26
    assert abs((res - ref) / ref) < 0.006