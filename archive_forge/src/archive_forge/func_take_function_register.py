from pyparsing import *
from sys import stdin, argv, exit
def take_function_register(self, rtype=SharedData.TYPES.NO_TYPE):
    """Reserves register for function return value and sets its type"""
    reg = SharedData.FUNCTION_REGISTER
    if reg not in self.free_registers:
        self.error('function register already taken')
    self.free_registers.remove(reg)
    self.used_registers.append(reg)
    self.symtab.set_type(reg, rtype)
    return reg