import re
from pygments.lexer import Lexer, RegexLexer, include, words, do_insertions
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \

    Pygments Lexer for R documentation (Rd) files

    This is a very minimal implementation, highlighting little more
    than the macros. A description of Rd syntax is found in `Writing R
    Extensions <http://cran.r-project.org/doc/manuals/R-exts.html>`_
    and `Parsing Rd files <developer.r-project.org/parseRd.pdf>`_.

    .. versionadded:: 1.6
    