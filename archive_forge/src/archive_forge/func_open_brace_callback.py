import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, default, \
from pygments.token import Name, Comment, String, Error, Number, Text, \
def open_brace_callback(self, match, ctx):
    opening_brace = match.group()
    ctx.opening_brace = opening_brace
    yield (match.start(), Punctuation, opening_brace)
    ctx.pos = match.end()