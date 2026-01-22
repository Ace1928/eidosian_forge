import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, LexerContext, \
from pygments.token import Text, Comment, Keyword, Name, String, Number, \
def parse_block_scalar_indent(token_class):
    """Process indentation spaces in a block scalar."""

    def callback(lexer, match, context):
        text = match.group()
        if context.block_scalar_indent is None:
            if len(text) <= max(context.indent, 0):
                context.stack.pop()
                context.stack.pop()
                return
            context.block_scalar_indent = len(text)
        elif len(text) < context.block_scalar_indent:
            context.stack.pop()
            context.stack.pop()
            return
        if text:
            yield (match.start(), token_class, text)
            context.pos = match.end()
    return callback