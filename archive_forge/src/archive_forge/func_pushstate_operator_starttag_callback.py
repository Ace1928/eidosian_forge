import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import unirange
from pygments.lexers.css import _indentation, _starts_block
from pygments.lexers.html import HtmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.ruby import RubyLexer
def pushstate_operator_starttag_callback(lexer, match, ctx):
    yield (match.start(), Name.Tag, match.group(1))
    lexer.xquery_parse_state.append('operator')
    ctx.stack.append('start_tag')
    ctx.pos = match.end()