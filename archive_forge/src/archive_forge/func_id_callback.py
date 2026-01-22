import re
from pygments.lexer import RegexLexer, include, bygroups, default, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def id_callback(self, match):
    str = match.group(1)
    if str in self.alphanumid_reserved:
        token = Keyword.Reserved
    elif str in self.symbolicid_reserved:
        token = Punctuation
    else:
        token = Name
    yield (match.start(1), token, str)