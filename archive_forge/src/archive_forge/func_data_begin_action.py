from pyparsing import *
from sys import stdin, argv, exit
def data_begin_action(self):
    """Inserts text at start of data segment"""
    self.codegen.prepare_data_segment()