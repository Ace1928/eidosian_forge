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
def test_long_args_command_split(self):
    self.do_layout_test('      # This very long command should be split to multiple lines\n      set(HEADERS very_long_header_name_a.h very_long_header_name_b.h very_long_header_name_c.h)\n      ', [(NodeType.BODY, 0, 0, 0, 63, [(NodeType.COMMENT, 0, 0, 0, 58, []), (NodeType.STATEMENT, 0, 1, 0, 63, [(NodeType.FUNNAME, 0, 1, 0, 3, []), (NodeType.LPAREN, 0, 1, 3, 4, []), (NodeType.ARGGROUP, 0, 1, 4, 63, [(NodeType.PARGGROUP, 0, 1, 4, 11, [(NodeType.ARGUMENT, 0, 1, 4, 11, [])]), (NodeType.PARGGROUP, 0, 1, 12, 63, [(NodeType.ARGUMENT, 0, 1, 12, 37, []), (NodeType.ARGUMENT, 0, 1, 38, 63, []), (NodeType.ARGUMENT, 0, 2, 12, 37, [])])]), (NodeType.RPAREN, 0, 2, 37, 38, [])])])])