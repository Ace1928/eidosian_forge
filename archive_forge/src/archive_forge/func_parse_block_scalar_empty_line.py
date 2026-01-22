import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, LexerContext, \
from pygments.token import Text, Comment, Keyword, Name, String, Number, \
def parse_block_scalar_empty_line(indent_token_class, content_token_class):
    """Process an empty line in a block scalar."""

    def callback(lexer, match, context):
        text = match.group()
        if context.block_scalar_indent is None or len(text) <= context.block_scalar_indent:
            if text:
                yield (match.start(), indent_token_class, text)
        else:
            indentation = text[:context.block_scalar_indent]
            content = text[context.block_scalar_indent:]
            yield (match.start(), indent_token_class, indentation)
            yield (match.start() + context.block_scalar_indent, content_token_class, content)
        context.pos = match.end()
    return callback