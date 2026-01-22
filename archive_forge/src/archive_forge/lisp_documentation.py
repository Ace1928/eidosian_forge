import re
from pygments.lexer import RegexLexer, include, bygroups, words, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.python import PythonLexer
An xtlang lexer for the `Extempore programming environment
    <http://extempore.moso.com.au>`_.

    This is a mixture of Scheme and xtlang, really. Keyword lists are
    taken from the Extempore Emacs mode
    (https://github.com/extemporelang/extempore-emacs-mode)

    .. versionadded:: 2.2
    