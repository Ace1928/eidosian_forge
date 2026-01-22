from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import io
import logging
import re
import sys
from cmakelang import lex
from cmakelang import markup
from cmakelang.common import UserError
from cmakelang.lex import TokenType
from cmakelang.parse.argument_nodes import PositionalGroupNode
from cmakelang.parse.common import FlowType, NodeType, TreeNode
from cmakelang.parse.util import comment_is_tag
from cmakelang.parse import simple_nodes
def assert_lt(self, cursor):
    assert self._cursor[0] < cursor[0] or (self._cursor[0] == cursor[0] and self._cursor[1] <= cursor[1]), 'self._cursor=({},{}), write_at=({}, {}):\n{}'.format(self._cursor[0], self._cursor[1], cursor[0], cursor[1], self.getvalue().encode('utf-8', errors='replace'))