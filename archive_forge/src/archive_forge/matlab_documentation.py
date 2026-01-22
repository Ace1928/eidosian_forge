import re
from pygments.lexer import Lexer, RegexLexer, bygroups, words, do_insertions
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers import _scilab_builtins

    For Scilab source code.

    .. versionadded:: 1.5
    