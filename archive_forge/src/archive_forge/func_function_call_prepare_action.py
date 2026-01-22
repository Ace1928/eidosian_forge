from pyparsing import *
from sys import stdin, argv, exit
def function_call_prepare_action(self, text, loc, fun):
    """Code executed after recognising a function call (type and function name)"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('FUN_PREP:', fun)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    index = self.symtab.lookup_symbol(fun.name, SharedData.KINDS.FUNCTION)
    if index == None:
        raise SemanticException("'%s' is not a function" % fun.name)
    self.function_call_stack.append(self.function_call_index)
    self.function_call_index = index
    self.function_arguments_stack.append(self.function_arguments[:])
    del self.function_arguments[:]
    self.codegen.save_used_registers()