from pygments.lexer import ExtendedRegexLexer, bygroups, DelegatingLexer
from pygments.token import Name, Number, String, Comment, Punctuation, \
def move_state(new_state):
    return ('#pop', new_state)