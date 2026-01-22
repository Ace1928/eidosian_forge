import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import unirange
from pygments.lexers.css import _indentation, _starts_block
from pygments.lexers.html import HtmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.ruby import RubyLexer
def pushstate_operator_processing_instruction_callback(lexer, match, ctx):
    yield (match.start(), String.Doc, match.group(1))
    ctx.stack.append('processing_instruction')
    lexer.xquery_parse_state.append('operator')
    ctx.pos = match.end()