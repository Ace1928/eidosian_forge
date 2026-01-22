from __future__ import print_function
import six
from patsy import PatsyError
from patsy.parse_formula import ParseNode, Token, parse_formula
from patsy.eval import EvalEnvironment, EvalFactor
from patsy.util import uniqueify_list
from patsy.util import repr_pretty_delegate, repr_pretty_impl
from patsy.util import no_pickling, assert_no_pickling
def test_eval_formula_error_reporting():
    from patsy.parse_formula import _parsing_error_test
    parse_fn = lambda formula: ModelDesc.from_formula(formula)
    _parsing_error_test(parse_fn, _eval_error_tests)