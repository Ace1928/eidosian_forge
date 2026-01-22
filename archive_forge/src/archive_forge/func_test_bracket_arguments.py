from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import lex
from cmakelang.lex import TokenType
def test_bracket_arguments(self):
    self.assert_tok_types('foo(bar [=[hello world]=] baz)', [TokenType.WORD, TokenType.LEFT_PAREN, TokenType.WORD, TokenType.WHITESPACE, TokenType.BRACKET_ARGUMENT, TokenType.WHITESPACE, TokenType.WORD, TokenType.RIGHT_PAREN])