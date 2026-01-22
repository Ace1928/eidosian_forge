from __future__ import unicode_literals
import contextlib
import logging
import unittest
import sys
from cmakelang.format import __main__
from cmakelang import configuration
from cmakelang import lex
from cmakelang import parse
from cmakelang.format import formatter
from cmakelang.parse.common import NodeType
def test_arg_just_fits(self):
    """
    Ensure that if an argument *just* fits that it isn't superfluously wrapped
    """
    with self.subTest(chars=81):
        self.do_layout_test('      message(FATAL_ERROR "81 character line ----------------------------------------")\n    ', [(NodeType.BODY, 0, 0, 0, 75, [(NodeType.STATEMENT, 1, 0, 0, 75, [(NodeType.FUNNAME, 0, 0, 0, 7, []), (NodeType.LPAREN, 0, 0, 7, 8, []), (NodeType.ARGGROUP, 0, 1, 2, 74, [(NodeType.KWARGGROUP, 0, 1, 2, 74, [(NodeType.KEYWORD, 0, 1, 2, 13, []), (NodeType.ARGGROUP, 0, 1, 14, 74, [(NodeType.PARGGROUP, 0, 1, 14, 74, [(NodeType.ARGUMENT, 0, 1, 14, 74, [])])])])]), (NodeType.RPAREN, 0, 1, 74, 75, [])])])])
    with self.subTest(chars=100, with_prefix=True):
        self.do_layout_test('      message(FATAL_ERROR\n              "100 character line ----------------------------------------------------------"\n      ) # Closing parenthesis is indented one space!\n    ', [(NodeType.BODY, 0, 0, 0, 83, [(NodeType.STATEMENT, 5, 0, 0, 83, [(NodeType.FUNNAME, 0, 0, 0, 7, []), (NodeType.LPAREN, 0, 0, 7, 8, []), (NodeType.ARGGROUP, 5, 1, 2, 83, [(NodeType.KWARGGROUP, 5, 1, 2, 83, [(NodeType.KEYWORD, 0, 1, 2, 13, []), (NodeType.ARGGROUP, 5, 2, 4, 83, [(NodeType.PARGGROUP, 4, 2, 4, 83, [(NodeType.ARGUMENT, 0, 2, 4, 83, [])])])])]), (NodeType.RPAREN, 0, 3, 0, 1, []), (NodeType.COMMENT, 0, 3, 2, 46, [])])])])
    with self.subTest(chars=100):
        self.do_layout_test('      message(\n        "100 character line ----------------------------------------------------------------------"\n        ) # Closing parenthesis is indented one space!\n    ', [(NodeType.BODY, 0, 0, 0, 93, [(NodeType.STATEMENT, 5, 0, 0, 93, [(NodeType.FUNNAME, 0, 0, 0, 7, []), (NodeType.LPAREN, 0, 0, 7, 8, []), (NodeType.ARGGROUP, 5, 1, 2, 93, [(NodeType.PARGGROUP, 4, 1, 2, 93, [(NodeType.ARGUMENT, 0, 1, 2, 93, [])])]), (NodeType.RPAREN, 0, 2, 0, 1, []), (NodeType.COMMENT, 0, 2, 2, 46, [])])])])