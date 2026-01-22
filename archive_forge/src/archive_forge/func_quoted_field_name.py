from pygments.lexer import include, RegexLexer, words
from pygments.token import Comment, Keyword, Name, Number, Operator, \
def quoted_field_name(quote_mark):
    return [('([^{quote}\\\\]|\\\\.)*{quote}'.format(quote=quote_mark), Name.Variable, 'field_separator')]