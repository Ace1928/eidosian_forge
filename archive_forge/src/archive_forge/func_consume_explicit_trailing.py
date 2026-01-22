from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.parse.util import (
from cmakelang.parse.common import NodeType, TreeNode
@classmethod
def consume_explicit_trailing(cls, ctx, tokens, parent):
    """
    Consume sequential comment lines, removing tokens from the input list and
    appending the resulting node as a child to the provided parent
    """
    while tokens and tokens[0].type in (lex.TokenType.WHITESPACE, lex.TokenType.NEWLINE):
        parent.children.append(tokens.pop(0))
    node = cls()
    node.is_explicit_trailing = True
    parent.children.append(node)
    while tokens and next_is_explicit_trailing_comment(ctx.config, tokens):
        node.children.append(tokens.pop(0))