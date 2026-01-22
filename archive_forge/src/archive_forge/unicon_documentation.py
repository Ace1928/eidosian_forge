import re
from pygments.lexer import RegexLexer, include, bygroups, words, using, this
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
endsuspend and endrepeat are unique to this language, and
        \self, /self doesn't seem to get used anywhere else either.