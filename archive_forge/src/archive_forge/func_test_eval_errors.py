from __future__ import print_function
import re
import six
import numpy as np
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (atleast_2d_column_default,
from patsy.infix_parser import Token, Operator, infix_parse
from patsy.parse_formula import _parsing_error_test
def test_eval_errors():

    def doit(bad_code):
        return linear_constraint(bad_code, ['a', 'b', 'c'])
    _parsing_error_test(doit, _parse_eval_error_tests)