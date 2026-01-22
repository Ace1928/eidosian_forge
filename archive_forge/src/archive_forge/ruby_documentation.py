import re
from pygments.lexer import Lexer, RegexLexer, ExtendedRegexLexer, include, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches

    Pygments Lexer For `Fancy <http://www.fancy-lang.org/>`_.

    Fancy is a self-hosted, pure object-oriented, dynamic,
    class-based, concurrent general-purpose programming language
    running on Rubinius, the Ruby VM.

    .. versionadded:: 1.5
    