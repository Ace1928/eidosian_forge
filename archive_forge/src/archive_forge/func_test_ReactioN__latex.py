from functools import reduce
from operator import attrgetter, add
import sys
from sympy import nsimplify
import pytest
from ..util.arithmeticdict import ArithmeticDict
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, to_unitless, allclose
from ..chemistry import (
@requires(parsing_library)
def test_ReactioN__latex():
    keys = 'H2O H2 O2'.split()
    subst = {k: Substance.from_formula(k) for k in keys}
    r2 = Reaction.from_string('2 H2O -> 2 H2 + O2', subst)
    assert r2.latex(subst) == '2 H_{2}O \\rightarrow 2 H_{2} + O_{2}'
    r3 = Reaction.from_string("2 H2O -> 2 H2 + O2; 42; name='split'", subst)
    assert r3.latex(subst, with_param=True, with_name=True) == '2 H_{2}O \\rightarrow 2 H_{2} + O_{2}; 42; split'
    assert r3.latex(subst, with_name=True) == '2 H_{2}O \\rightarrow 2 H_{2} + O_{2}; split'
    assert r3.latex(subst, with_param=True) == '2 H_{2}O \\rightarrow 2 H_{2} + O_{2}; 42'
    assert r3.latex(subst) == '2 H_{2}O \\rightarrow 2 H_{2} + O_{2}'