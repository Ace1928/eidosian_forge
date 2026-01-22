from pyparsing import *
from sys import stdin, argv, exit
def mulexp_action(self, text, loc, mul):
    """Code executed after recognising a mulexp expression (something *|/ something)"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('MUL_EXP:', mul)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    m = list(mul)
    while len(m) > 1:
        if not self.symtab.same_types(m[0], m[2]):
            raise SemanticException("Invalid opernads to binary '%s'" % m[1])
        reg = self.codegen.arithmetic(m[1], m[0], m[2])
        m[0:3] = [reg]
    return m[0]