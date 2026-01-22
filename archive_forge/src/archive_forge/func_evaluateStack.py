from pyparsing import ParseException, Word, alphas, alphanums
import math
from fourFn import BNF, exprStack, fn, opn
def evaluateStack(s):
    op = s.pop()
    if op == 'unary -':
        return -evaluateStack(s)
    if op in '+-*/^':
        op2 = evaluateStack(s)
        op1 = evaluateStack(s)
        return opn[op](op1, op2)
    elif op == 'PI':
        return math.pi
    elif op == 'E':
        return math.e
    elif op in fn:
        return fn[op](evaluateStack(s))
    elif op[0].isalpha():
        if op in variables:
            return variables[op]
        raise Exception("invalid identifier '%s'" % op)
    else:
        return float(op)