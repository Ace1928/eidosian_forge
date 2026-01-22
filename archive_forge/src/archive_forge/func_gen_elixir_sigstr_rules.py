import re
from pygments.lexer import Lexer, RegexLexer, bygroups, words, do_insertions, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def gen_elixir_sigstr_rules(term, token, interpol=True):
    if interpol:
        return [('[^#%s\\\\]+' % (term,), token), include('escapes'), ('\\\\.', token), ('%s[a-zA-Z]*' % (term,), token, '#pop'), include('interpol')]
    else:
        return [('[^%s\\\\]+' % (term,), token), ('\\\\.', token), ('%s[a-zA-Z]*' % (term,), token, '#pop')]