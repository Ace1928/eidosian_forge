import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, LexerContext, \
from pygments.token import Text, Comment, Keyword, Name, String, Number, \
def set_indent(token_class, implicit=False):
    """Set the previously saved indentation level."""

    def callback(lexer, match, context):
        text = match.group()
        if context.indent < context.next_indent:
            context.indent_stack.append(context.indent)
            context.indent = context.next_indent
        if not implicit:
            context.next_indent += len(text)
        yield (match.start(), token_class, text)
        context.pos = match.end()
    return callback