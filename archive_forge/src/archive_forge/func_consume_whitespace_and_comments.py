from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.parse.util import (
from cmakelang.parse.common import NodeType, TreeNode
def consume_whitespace_and_comments(ctx, tokens, tree):
    """Consume any whitespace or comments that occur at the current depth
  """
    while tokens:
        if tokens[0].type in WHITESPACE_TOKENS:
            tree.children.append(tokens.pop(0))
            continue
        if tokens[0].type in (lex.TokenType.COMMENT, lex.TokenType.BRACKET_COMMENT):
            child = CommentNode.consume(ctx, tokens)
            tree.children.append(child)
            continue
        if tokens[0].type in (lex.TokenType.FORMAT_OFF, lex.TokenType.FORMAT_ON):
            tree.children.append(OnOffNode.consume(ctx, tokens))
            continue
        break