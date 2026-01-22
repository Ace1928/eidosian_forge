from __future__ import unicode_literals
import unittest
from cmakelang import configuration
from cmakelang import lex
from cmakelang import parse
from cmakelang.parse.printer import tree_string, test_string
from cmakelang.parse.common import NodeType
def test_shellcommand_parse(self):
    self.do_type_test('      add_test(NAME foo-test\n               COMMAND cmdname -Bm -h --hello foo bar --world baz buck\n               WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})\n      ', [(NodeType.BODY, [(NodeType.WHITESPACE, []), (NodeType.STATEMENT, [(NodeType.FUNNAME, []), (NodeType.LPAREN, []), (NodeType.ARGGROUP, [(NodeType.KWARGGROUP, [(NodeType.KEYWORD, []), (NodeType.ARGGROUP, [(NodeType.PARGGROUP, [(NodeType.ARGUMENT, [])])])]), (NodeType.KWARGGROUP, [(NodeType.KEYWORD, []), (NodeType.ARGGROUP, [(NodeType.PARGGROUP, [(NodeType.ARGUMENT, []), (NodeType.ARGUMENT, []), (NodeType.ARGUMENT, []), (NodeType.ARGUMENT, []), (NodeType.ARGUMENT, []), (NodeType.ARGUMENT, []), (NodeType.ARGUMENT, []), (NodeType.ARGUMENT, []), (NodeType.ARGUMENT, [])])])]), (NodeType.KWARGGROUP, [(NodeType.KEYWORD, []), (NodeType.ARGGROUP, [(NodeType.PARGGROUP, [(NodeType.ARGUMENT, [])])])])]), (NodeType.RPAREN, [])]), (NodeType.WHITESPACE, [])])])