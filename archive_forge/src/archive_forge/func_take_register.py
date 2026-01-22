from pyparsing import *
from sys import stdin, argv, exit
def take_register(self, rtype=SharedData.TYPES.NO_TYPE):
    """Reserves one working register and sets its type"""
    if len(self.free_registers) == 0:
        self.error('no more free registers')
    reg = self.free_registers.pop()
    self.used_registers.append(reg)
    self.symtab.set_type(reg, rtype)
    return reg