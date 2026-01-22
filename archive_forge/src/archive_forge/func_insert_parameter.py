from pyparsing import *
from sys import stdin, argv, exit
def insert_parameter(self, pname, ptype):
    """Inserts a new parameter"""
    index = self.insert_id(pname, SharedData.KINDS.PARAMETER, SharedData.KINDS.PARAMETER, ptype)
    self.table[index].set_attribute('Index', self.shared.function_params)
    self.table[self.shared.function_index].param_types.append(ptype)
    return index