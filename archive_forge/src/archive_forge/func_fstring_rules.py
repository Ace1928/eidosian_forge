import re
import keyword
from pygments.lexer import DelegatingLexer, Lexer, RegexLexer, include, \
from pygments.util import get_bool_opt, shebang_matches
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments import unistring as uni
def fstring_rules(ttype):
    return [('\\}', String.Interpol), ('\\{', String.Interpol, 'expr-inside-fstring'), ('[^\\\\\\\'"{}\\n]+', ttype), ('[\\\'"\\\\]', ttype)]