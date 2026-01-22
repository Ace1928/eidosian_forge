import re
from pygments.lexer import RegexLexer, include, bygroups, words
from pygments.token import Comment, Keyword, Name, Number, Operator, \
from pygments.lexers._qlik_builtins import OPERATORS_LIST, STATEMENT_LIST, \

    Lexer for qlik code, including .qvs files

    .. versionadded:: 2.12
    