from __future__ import print_function
import re
import six
import numpy as np
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (atleast_2d_column_default,
from patsy.infix_parser import Token, Operator, infix_parse
from patsy.parse_formula import _parsing_error_test
def test__tokenize_constraint():
    code = '2 * (a + b) = q'
    tokens = _tokenize_constraint(code, ['a', 'b', 'q'])
    expecteds = [('NUMBER', 0, 1, '2'), ('*', 2, 3, '*'), (Token.LPAREN, 4, 5, '('), ('VARIABLE', 5, 6, 'a'), ('+', 7, 8, '+'), ('VARIABLE', 9, 10, 'b'), (Token.RPAREN, 10, 11, ')'), ('=', 12, 13, '='), ('VARIABLE', 14, 15, 'q')]
    for got, expected in zip(tokens, expecteds):
        assert isinstance(got, Token)
        assert got.type == expected[0]
        assert got.origin == Origin(code, expected[1], expected[2])
        assert got.extra == expected[3]
    import pytest
    pytest.raises(PatsyError, _tokenize_constraint, '1 + @b', ['b'])
    _tokenize_constraint('1 + @b', ['@b'])
    for names in (['a', 'aa'], ['aa', 'a']):
        tokens = _tokenize_constraint('a aa a', names)
        assert len(tokens) == 3
        assert [t.extra for t in tokens] == ['a', 'aa', 'a']
    tokens = _tokenize_constraint('2 * a[1,1],', ['a[1,1]'])
    assert len(tokens) == 4
    assert [t.type for t in tokens] == ['NUMBER', '*', 'VARIABLE', ',']
    assert [t.extra for t in tokens] == ['2', '*', 'a[1,1]', ',']