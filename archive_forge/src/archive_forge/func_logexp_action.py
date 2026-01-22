from pyparsing import *
from sys import stdin, argv, exit
def logexp_action(self, text, loc, arg):
    """Code executed after recognising logexp expression (something or something)"""
    exshared.setpos(loc, text)
    if DEBUG > 0:
        print('LOG_EXP:', arg)
        if DEBUG == 2:
            self.symtab.display()
        if DEBUG > 2:
            return
    label = self.codegen.label('true{0}'.format(self.label_number), True, False)
    self.codegen.jump(self.relexp_code, False, label)
    self.codegen.newline_label('false{0}'.format(self.false_label_number), True, True)
    self.false_label_number += 1