from pyparsing import *
from sys import stdin, argv, exit
def return_action(self, text, loc, ret):
    """Code executed after recognising a return statement"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('RETURN:', ret)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    if not self.symtab.same_types(self.shared.function_index, ret.exp[0]):
        raise SemanticException('Incompatible type in return')
    reg = self.codegen.take_function_register()
    self.codegen.move(ret.exp[0], reg)
    self.codegen.free_register(reg)
    self.codegen.unconditional_jump(self.codegen.label(self.shared.function_name + '_exit', True))