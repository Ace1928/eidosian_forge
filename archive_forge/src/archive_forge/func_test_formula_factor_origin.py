from __future__ import print_function
import six
from patsy import PatsyError
from patsy.parse_formula import ParseNode, Token, parse_formula
from patsy.eval import EvalEnvironment, EvalFactor
from patsy.util import uniqueify_list
from patsy.util import repr_pretty_delegate, repr_pretty_impl
from patsy.util import no_pickling, assert_no_pickling
def test_formula_factor_origin():
    from patsy.origin import Origin
    desc = ModelDesc.from_formula('a + b')
    assert desc.rhs_termlist[1].factors[0].origin == Origin('a + b', 0, 1)
    assert desc.rhs_termlist[2].factors[0].origin == Origin('a + b', 4, 5)