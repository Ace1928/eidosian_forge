import re
from pygments.lexer import RegexLexer, include, bygroups, using, words, \
from pygments.lexers.c_cpp import CppLexer, CLexer
from pygments.lexers.d import DLexer
from pygments.token import Text, Name, Number, String, Comment, Punctuation, \
def guess_identifier(lexer, match):
    ident = match.group(0)
    klass = Name.Variable if ident.upper() in lexer.REGISTERS else Name.Label
    yield (match.start(), klass, ident)