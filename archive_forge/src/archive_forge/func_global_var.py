from pyparsing import *
from sys import stdin, argv, exit
def global_var(self, name):
    """Inserts a new static (global) variable definition"""
    self.newline_label(name, False, True)
    self.newline_text('WORD\t1', True)