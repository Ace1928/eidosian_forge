from pyparsing import *
from sys import stdin, argv, exit
def function_call(self, function, arguments):
    """Generates code for a function call
           function - function index in symbol table
           arguments - list of arguments (indexes in symbol table)
        """
    for arg in arguments:
        self.push(self.symbol(arg))
        self.free_if_register(arg)
    self.newline_text('CALL\t' + self.symtab.get_name(function), True)
    args = self.symtab.get_attribute(function)
    if args > 0:
        args_space = self.symtab.insert_constant('{0}'.format(args * 4), SharedData.TYPES.UNSIGNED)
        self.arithmetic('+', '%15', args_space, '%15')