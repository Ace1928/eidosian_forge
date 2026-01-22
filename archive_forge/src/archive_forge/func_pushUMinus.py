from pyparsing import Literal,Word,Group,\
import math
import operator
def pushUMinus(strg, loc, toks):
    for t in toks:
        if t == '-':
            exprStack.append('unary -')
        else:
            break