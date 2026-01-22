from pyparsing import *
from sys import stdin, argv, exit
def function_begin_action(self, text, loc, fun):
    """Code executed after recognising a function definition (type and function name)"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('FUN_BEGIN:', fun)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    self.shared.function_index = self.symtab.insert_function(fun.name, fun.type)
    self.shared.function_name = fun.name
    self.shared.function_params = 0
    self.shared.function_vars = 0
    self.codegen.function_begin()