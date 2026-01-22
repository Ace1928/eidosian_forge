import logging
from cmakelang.lex import TokenType
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.common import KwargBreaker, NodeType, TreeNode
from cmakelang.parse.util import (
def parse_add_custom_command(ctx, tokens, breakstack):
    """
  There are two forms of `add_custom_command`. This is the dispatcher between
  the two forms.
  """
    descriminator_token = get_first_semantic_token(tokens)
    if descriminator_token is None or descriminator_token.type is TokenType.RIGHT_PAREN:
        location = ()
        if tokens:
            location = tokens[0].get_location()
        ctx.lint_ctx.record_lint('E1120', location=location)
        return StandardArgTree.parse(ctx, tokens, npargs='*', kwargs={}, flags=[], breakstack=breakstack)
    if descriminator_token.type is TokenType.DEREF:
        ctx.lint_ctx.record_lint('C0114', location=descriminator_token.get_location())
        return parse_add_custom_command_standard(ctx, tokens, breakstack)
    descriminator_word = get_normalized_kwarg(descriminator_token)
    if descriminator_word == 'TARGET':
        return parse_add_custom_command_events(ctx, tokens, breakstack)
    if descriminator_word == 'OUTPUT':
        return parse_add_custom_command_standard(ctx, tokens, breakstack)
    ctx.lint_ctx.record_lint('E1126', location=descriminator_token.get_location())
    return parse_add_custom_command_standard(ctx, tokens, breakstack)