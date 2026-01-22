from pyparsing import *
from sys import stdin, argv, exit
def if_else_action(self, text, loc, arg):
    """Code executed after recognising if statement's else body"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('IF_ELSE:', arg)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    self.label_number = self.label_stack.pop()
    label = self.codegen.label('exit{0}'.format(self.label_number), True, False)
    self.codegen.unconditional_jump(label)
    self.codegen.newline_label('false{0}'.format(self.label_stack.pop()), True, True)
    self.label_stack.append(self.label_number)