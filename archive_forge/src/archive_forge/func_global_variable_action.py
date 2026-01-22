from pyparsing import *
from sys import stdin, argv, exit
def global_variable_action(self, text, loc, var):
    """Code executed after recognising a global variable"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('GLOBAL_VAR:', var)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    index = self.symtab.insert_global_var(var.name, var.type)
    self.codegen.global_var(var.name)
    return index