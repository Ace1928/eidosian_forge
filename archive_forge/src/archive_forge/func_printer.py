from pyparsing import Literal, CaselessLiteral, Word, delimitedList \
def printer(s, loc, tok):
    print(tok, end=' ')
    return tok