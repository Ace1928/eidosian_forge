import re
from pygments.lexer import Lexer, RegexLexer, ExtendedRegexLexer, include, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
def intp_string_callback(self, match, ctx):
    yield (match.start(1), String.Other, match.group(1))
    nctx = LexerContext(match.group(3), 0, ['interpolated-string'])
    for i, t, v in self.get_tokens_unprocessed(context=nctx):
        yield (match.start(3) + i, t, v)
    yield (match.start(4), String.Other, match.group(4))
    ctx.pos = match.end()