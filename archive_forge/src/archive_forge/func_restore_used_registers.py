from pyparsing import *
from sys import stdin, argv, exit
def restore_used_registers(self):
    """Pops all used working registers after function call"""
    used = self.used_registers_stack.pop()
    self.used_registers = used[:]
    used.sort(reverse=True)
    for reg in used:
        self.newline_text('POP \t%s' % SharedData.REGISTERS[reg], True)
        self.free_registers.remove(reg)