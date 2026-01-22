import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import unirange
from pygments.lexers.css import _indentation, _starts_block
from pygments.lexers.html import HtmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.ruby import RubyLexer
def pushstate_operator_root_callback(lexer, match, ctx):
    yield (match.start(), Punctuation, match.group(1))
    lexer.xquery_parse_state.append('operator')
    ctx.stack = ['root']
    ctx.pos = match.end()