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
def test_sortable_set(self):
    self.do_layout_test('      set(SOURCES #[[cmf:sortable]] foo.cc bar.cc baz.cc)\n      ', [(NodeType.BODY, 0, 0, 0, 51, [(NodeType.STATEMENT, 0, 0, 0, 51, [(NodeType.FUNNAME, 0, 0, 0, 3, []), (NodeType.LPAREN, 0, 0, 3, 4, []), (NodeType.ARGGROUP, 0, 0, 4, 50, [(NodeType.PARGGROUP, 0, 0, 4, 11, [(NodeType.ARGUMENT, 0, 0, 4, 11, [])]), (NodeType.PARGGROUP, 0, 0, 12, 50, [(NodeType.COMMENT, 0, 0, 12, 29, []), (NodeType.ARGUMENT, 0, 0, 30, 36, []), (NodeType.ARGUMENT, 0, 0, 37, 43, []), (NodeType.ARGUMENT, 0, 0, 44, 50, [])])]), (NodeType.RPAREN, 0, 0, 50, 51, [])])])])