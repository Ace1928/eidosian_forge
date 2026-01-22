from pyparsing import *
from sys import stdin, argv, exit
def if_body_action(self, text, loc, arg):
    """Code executed after recognising if statement's body"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('IF_BODY:', arg)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    label = self.codegen.label('false{0}'.format(self.false_label_number), True, False)
    self.codegen.jump(self.relexp_code, True, label)
    self.codegen.newline_label('true{0}'.format(self.label_number), True, True)
    self.label_stack.append(self.false_label_number)
    self.label_stack.append(self.label_number)