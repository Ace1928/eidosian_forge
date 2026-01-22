from pyparsing import *
from sys import stdin, argv, exit
def numexp_action(self, text, loc, num):
    """Code executed after recognising a numexp expression (something +|- something)"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('NUM_EXP:', num)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    n = list(num)
    while len(n) > 1:
        if not self.symtab.same_types(n[0], n[2]):
            raise SemanticException("Invalid opernads to binary '%s'" % n[1])
        reg = self.codegen.arithmetic(n[1], n[0], n[2])
        n[0:3] = [reg]
    return n[0]