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
def test_Reaction__html():
    keys = 'H2O H2 O2'.split()
    subst = {k: Substance.from_formula(k) for k in keys}
    r2 = Reaction.from_string('2 H2O -> 2 H2 + O2', subst)
    assert r2.html(subst) == '2 H<sub>2</sub>O &rarr; 2 H<sub>2</sub> + O<sub>2</sub>'
    assert r2.html(subst, Reaction_coeff_fmt=lambda s: '<b>{0}</b>'.format(s)) == '<b>2</b> H<sub>2</sub>O &rarr; <b>2</b> H<sub>2</sub> + O<sub>2</sub>'
    assert r2.html(subst, Reaction_formula_fmt=lambda s: '<b>{0}</b>'.format(s)) == '2 <b>H<sub>2</sub>O</b> &rarr; 2 <b>H<sub>2</sub></b> + <b>O<sub>2</sub></b>'