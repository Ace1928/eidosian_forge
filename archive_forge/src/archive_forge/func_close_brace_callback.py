import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, default, \
from pygments.token import Name, Comment, String, Error, Number, Text, \
def close_brace_callback(self, match, ctx):
    closing_brace = match.group()
    if ctx.opening_brace == '{' and closing_brace != '}' or (ctx.opening_brace == '(' and closing_brace != ')'):
        yield (match.start(), Error, closing_brace)
    else:
        yield (match.start(), Punctuation, closing_brace)
    del ctx.opening_brace
    ctx.pos = match.end()