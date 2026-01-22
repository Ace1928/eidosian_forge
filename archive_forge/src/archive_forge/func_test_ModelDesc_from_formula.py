from __future__ import print_function
import six
from patsy import PatsyError
from patsy.parse_formula import ParseNode, Token, parse_formula
from patsy.eval import EvalEnvironment, EvalFactor
from patsy.util import uniqueify_list
from patsy.util import repr_pretty_delegate, repr_pretty_impl
from patsy.util import no_pickling, assert_no_pickling
def test_ModelDesc_from_formula():
    for input in ('y ~ x', parse_formula('y ~ x')):
        md = ModelDesc.from_formula(input)
        assert md.lhs_termlist == [Term([EvalFactor('y')])]
        assert md.rhs_termlist == [INTERCEPT, Term([EvalFactor('x')])]