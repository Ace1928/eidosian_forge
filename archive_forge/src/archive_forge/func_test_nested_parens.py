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
def test_nested_parens(self):
    self.do_layout_test('      if((NOT HELLO) OR (NOT EXISTS ${WORLD}))\n        message(WARNING "something is wrong")\n        set(foobar FALSE)\n      endif()\n      ', [(NodeType.BODY, 0, 0, 0, 40, [(NodeType.FLOW_CONTROL, 0, 0, 0, 40, [(NodeType.STATEMENT, 0, 0, 0, 40, [(NodeType.FUNNAME, 0, 0, 0, 2, []), (NodeType.LPAREN, 0, 0, 2, 3, []), (NodeType.ARGGROUP, 0, 0, 3, 39, [(NodeType.PARENGROUP, 0, 0, 3, 14, [(NodeType.LPAREN, 0, 0, 3, 4, []), (NodeType.ARGGROUP, 0, 0, 4, 13, [(NodeType.PARGGROUP, 0, 0, 4, 13, [(NodeType.FLAG, 0, 0, 4, 7, []), (NodeType.ARGUMENT, 0, 0, 8, 13, [])])]), (NodeType.RPAREN, 0, 0, 13, 14, [])]), (NodeType.KWARGGROUP, 0, 0, 15, 39, [(NodeType.KEYWORD, 0, 0, 15, 17, []), (NodeType.ARGGROUP, 0, 0, 18, 39, [(NodeType.PARENGROUP, 0, 0, 18, 39, [(NodeType.LPAREN, 0, 0, 18, 19, []), (NodeType.ARGGROUP, 0, 0, 19, 38, [(NodeType.PARGGROUP, 0, 0, 19, 38, [(NodeType.FLAG, 0, 0, 19, 22, []), (NodeType.FLAG, 0, 0, 23, 29, []), (NodeType.ARGUMENT, 0, 0, 30, 38, [])])]), (NodeType.RPAREN, 0, 0, 38, 39, [])])])])]), (NodeType.RPAREN, 0, 0, 39, 40, [])]), (NodeType.BODY, 0, 1, 2, 39, [(NodeType.STATEMENT, 0, 1, 2, 39, [(NodeType.FUNNAME, 0, 1, 2, 9, []), (NodeType.LPAREN, 0, 1, 9, 10, []), (NodeType.ARGGROUP, 0, 1, 10, 38, [(NodeType.KWARGGROUP, 0, 1, 10, 38, [(NodeType.KEYWORD, 0, 1, 10, 17, []), (NodeType.ARGGROUP, 0, 1, 18, 38, [(NodeType.PARGGROUP, 0, 1, 18, 38, [(NodeType.ARGUMENT, 0, 1, 18, 38, [])])])])]), (NodeType.RPAREN, 0, 1, 38, 39, [])]), (NodeType.STATEMENT, 0, 2, 2, 19, [(NodeType.FUNNAME, 0, 2, 2, 5, []), (NodeType.LPAREN, 0, 2, 5, 6, []), (NodeType.ARGGROUP, 0, 2, 6, 18, [(NodeType.PARGGROUP, 0, 2, 6, 12, [(NodeType.ARGUMENT, 0, 2, 6, 12, [])]), (NodeType.PARGGROUP, 0, 2, 13, 18, [(NodeType.ARGUMENT, 0, 2, 13, 18, [])])]), (NodeType.RPAREN, 0, 2, 18, 19, [])])]), (NodeType.STATEMENT, 0, 3, 0, 7, [(NodeType.FUNNAME, 0, 3, 0, 5, []), (NodeType.LPAREN, 0, 3, 5, 6, []), (NodeType.ARGGROUP, 0, 3, 6, 6, []), (NodeType.RPAREN, 0, 3, 6, 7, [])])])])])