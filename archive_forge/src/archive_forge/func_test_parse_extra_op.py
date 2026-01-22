from __future__ import print_function
import tokenize
import six
from six.moves import cStringIO as StringIO
from patsy import PatsyError
from patsy.origin import Origin
from patsy.infix_parser import Token, Operator, infix_parse, ParseNode
from patsy.tokens import python_tokenize, pretty_untokenize
from patsy.util import PushbackAdapter
def test_parse_extra_op():
    extra_operators = [Operator('|', 2, 250)]
    _do_parse_test(_parser_tests, extra_operators=extra_operators)
    _do_parse_test(_extra_op_parser_tests, extra_operators=extra_operators)
    test_parse_errors(extra_operators=extra_operators)