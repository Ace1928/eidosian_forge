import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
def indentify(s):
    out = []
    stack = []
    l = ['', s]
    for tokenType, tokenString, (startRow, startColumn), (endRow, endColumn), logicalLine in tokenize(l.pop):
        if tokenString in ['[', '(', '{']:
            stack.append(tokenString)
        elif tokenString in [']', ')', '}']:
            stack.pop()
        if tokenString == '\x00':
            out.append('  ' * len(stack))
        else:
            out.append(tokenString)
    return ''.join(out)