import pickle
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.utilities.matchpy_connector import WildDot, WildPlus, WildStar, Replacer
def test_matchpy_object_pickle():
    if matchpy is None:
        return
    a1 = WildDot('a')
    a2 = pickle.loads(pickle.dumps(a1))
    assert a1 == a2
    a1 = WildDot('a', S(1))
    a2 = pickle.loads(pickle.dumps(a1))
    assert a1 == a2
    a1 = WildPlus('a', S(1))
    a2 = pickle.loads(pickle.dumps(a1))
    assert a1 == a2
    a1 = WildStar('a', S(1))
    a2 = pickle.loads(pickle.dumps(a1))
    assert a1 == a2