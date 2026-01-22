import re
from pygments.lexer import Lexer, RegexLexer, bygroups, words, do_insertions, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def gen_elixir_string_rules(name, symbol, token):
    states = {}
    states['string_' + name] = [('[^#%s\\\\]+' % (symbol,), token), include('escapes'), ('\\\\.', token), ('(%s)' % (symbol,), bygroups(token), '#pop'), include('interpol')]
    return states