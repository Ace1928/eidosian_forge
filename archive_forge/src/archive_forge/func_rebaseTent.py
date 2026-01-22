from fontTools.varLib.models import supportScalar
from fontTools.misc.fixedTools import MAX_F2DOT14
from functools import lru_cache
@lru_cache(128)
def rebaseTent(tent, axisLimit):
    """Given a tuple (lower,peak,upper) "tent" and new axis limits
    (axisMin,axisDefault,axisMax), solves how to represent the tent
    under the new axis configuration.  All values are in normalized
    -1,0,+1 coordinate system. Tent values can be outside this range.

    Return value is a list of tuples. Each tuple is of the form
    (scalar,tent), where scalar is a multipler to multiply any
    delta-sets by, and tent is a new tent for that output delta-set.
    If tent value is None, that is a special deltaset that should
    be always-enabled (called "gain")."""
    axisMin, axisDef, axisMax, _distanceNegative, _distancePositive = axisLimit
    assert -1 <= axisMin <= axisDef <= axisMax <= +1
    lower, peak, upper = tent
    assert -2 <= lower <= peak <= upper <= +2
    assert peak != 0
    sols = _solve(tent, axisLimit)
    n = lambda v: axisLimit.renormalizeValue(v)
    sols = [(scalar, (n(v[0]), n(v[1]), n(v[2])) if v is not None else None) for scalar, v in sols if scalar]
    return sols