from __future__ import print_function
import tokenize
import six
from six.moves import cStringIO as StringIO
from patsy import PatsyError
from patsy.origin import Origin
from patsy.infix_parser import Token, Operator, infix_parse, ParseNode
from patsy.tokens import python_tokenize, pretty_untokenize
from patsy.util import PushbackAdapter
def test_parse_errors(extra_operators=[]):

    def parse_fn(code):
        return parse_formula(code, extra_operators=extra_operators)
    _parsing_error_test(parse_fn, _parser_error_tests)