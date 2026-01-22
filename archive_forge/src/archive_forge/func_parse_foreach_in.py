import logging
from cmakelang import lex
from cmakelang.parse.common import KwargBreaker, NodeType
from cmakelang.parse.common import TreeNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_foreach_in(ctx, tokens, breakstack):
    """
  ::

    foreach(loop_var IN [LISTS [<lists>]] [ITEMS [<items>]])
  """
    tree = ArgGroupNode()
    while tokens and tokens[0].type in WHITESPACE_TOKENS:
        tree.children.append(tokens.pop(0))
        continue
    ntokens = len(tokens)
    pargs = PositionalGroupNode.parse(ctx, tokens, 2, ['IN'], breakstack)
    assert len(tokens) < ntokens
    tree.children.append(pargs)
    kwargs = {'LISTS': PositionalParser('+'), 'ITEMS': PositionalParser('+')}
    sub_breakstack = breakstack + [KwargBreaker(list(kwargs.keys()))]
    while tokens:
        if should_break(tokens[0], breakstack):
            break
        if tokens[0].type in WHITESPACE_TOKENS:
            tree.children.append(tokens.pop(0))
            continue
        if tokens[0].type in (lex.TokenType.COMMENT, lex.TokenType.BRACKET_COMMENT):
            if not get_tag(tokens[0]) in ('sort', 'sortable'):
                child = TreeNode(NodeType.COMMENT)
                tree.children.append(child)
                child.children.append(tokens.pop(0))
                continue
        ntokens = len(tokens)
        word = get_normalized_kwarg(tokens[0])
        subparser = kwargs.get(word, None)
        if subparser is not None:
            subtree = KeywordGroupNode.parse(ctx, tokens, word, subparser, sub_breakstack)
        else:
            logger.warning('Unexpected positional argument at %s', tokens[0].location)
            subtree = PositionalGroupNode.parse(ctx, tokens, '+', [], sub_breakstack)
        assert len(tokens) < ntokens
        tree.children.append(subtree)
    return tree