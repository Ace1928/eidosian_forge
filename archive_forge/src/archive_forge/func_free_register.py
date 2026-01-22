from pyparsing import *
from sys import stdin, argv, exit
def free_register(self, reg):
    """Releases working register"""
    if reg not in self.used_registers:
        self.error('register %s is not taken' % self.REGISTERS[reg])
    self.used_registers.remove(reg)
    self.free_registers.append(reg)
    self.free_registers.sort(reverse=True)