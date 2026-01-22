from pyparsing import *
from sys import stdin, argv, exit
def relop_code(self, relop, operands_type):
    """Returns code for relational operator
           relop - relational operator
           operands_type - int or unsigned
        """
    code = self.RELATIONAL_DICT[relop]
    offset = 0 if operands_type == SharedData.TYPES.INT else len(SharedData.RELATIONAL_OPERATORS)
    return code + offset