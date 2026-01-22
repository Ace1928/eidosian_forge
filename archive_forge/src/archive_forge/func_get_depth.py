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
def get_depth(self):
    """
    Compute and return the depth of the subtree rooted at this node. The
    depth of the tree is the depth of the deepest (leaf) descendant.
    """
    if self._children:
        return 1 + max((child.get_depth() for child in self._children))
    return 1