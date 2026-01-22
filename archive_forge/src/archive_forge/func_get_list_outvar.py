import enum
import logging
import re
from cmakelang.common import InternalError
from cmakelang.format.formatter import get_comment_lines
from cmakelang.lex import TokenType, Token
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.body_nodes import BodyNode, FlowControlNode
from cmakelang.parse.simple_nodes import CommentNode
from cmakelang.parse.common import NodeType, TreeNode
from cmakelang.parse.statement_node import StatementNode
from cmakelang.parse.util import get_min_npargs
from cmakelang.parse import variables
from cmakelang.parse.funs.set import SetFnNode
def get_list_outvar(node):
    """Given a statement parse node for a `list()` command, inspect the
     subcommand and extract the token corresponding to the output
     variable name."""
    semtoks = node.argtree.parg_groups[0].get_tokens(kind='semantic')
    listcmd = semtoks[0].spelling
    if listcmd.upper() in ('LENGTH', 'GET', 'JOIN', 'SUBLIST'):
        return semtoks[-1]
    return None