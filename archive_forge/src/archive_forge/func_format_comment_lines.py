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
def format_comment_lines(node, stack_context, line_width):
    """
  Reflow comment lines into the given line width, parsing markup as necessary.
  """
    config = stack_context.config
    inlines = get_comment_lines(config, node)
    if isinstance(node, simple_nodes.CommentNode) and node.is_explicit_trailing:
        prefix = config.markup.explicit_trailing_pattern
    else:
        prefix = '#'
    if not config.markup.enable_markup:
        return [prefix + line.rstrip() for line in inlines]
    if config.markup.literal_comment_pattern is not None:
        literal_comment_regex = re.compile(config.markup.literal_comment_pattern)
        if literal_comment_regex.match('\n'.join(inlines)):
            return [prefix + line.rstrip() for line in inlines]
    if node.children[0] is stack_context.first_token and (config.markup.first_comment_is_literal or stack_context.first_token.spelling.startswith('#!')):
        return [prefix + line.rstrip() for line in inlines]
    items = markup.parse(inlines, config)
    markup_lines = markup.format_items(config, max(10, line_width - 2), items)
    return [prefix + ' ' * len(line[:1]) + line for line in markup_lines]