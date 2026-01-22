from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.parse.util import (
from cmakelang.parse.common import NodeType, TreeNode
@classmethod
def consume_implicit_trailing(cls, ctx, tokens, parent):
    """
    Consume sequential comment lines, removing tokens from the input list and
    appending the resulting node as a child to the provided parent
    """
    if tokens[0].type == lex.TokenType.WHITESPACE:
        parent.children.append(tokens.pop(0))
    node = cls()
    node.is_implicit_trailing = True
    parent.children.append(node)
    comment_tokens = node.children
    comment_tokens = []
    while tokens and is_valid_trailing_comment(tokens[0]):
        if comment_tokens and (not are_column_aligned(comment_tokens[-1], tokens[0])):
            break
        comment_token = tokens.pop(0)
        comment_tokens.append(comment_token)
        node.children.append(comment_token)
        if len(tokens) > 1 and tokens[0].type == lex.TokenType.NEWLINE and is_valid_trailing_comment(tokens[1]):
            node.children.append(tokens.pop(0))
        elif len(tokens) > 2 and tokens[0].type == lex.TokenType.NEWLINE and (tokens[1].type == lex.TokenType.WHITESPACE) and is_valid_trailing_comment(tokens[2]):
            node.children.append(tokens.pop(0))
            node.children.append(tokens.pop(0))