from sympy.external.gmpy import GROUND_TYPES
from sympy.external.importtools import version_tuple
from sympy.interactive.printing import init_printing
from sympy.utilities.misc import ARCH
from sympy import *
def enable_automatic_int_sympification(shell):
    """
    Allow IPython to automatically convert integer literals to Integer.
    """
    import ast
    old_run_cell = shell.run_cell

    def my_run_cell(cell, *args, **kwargs):
        try:
            ast.parse(cell)
        except SyntaxError:
            pass
        else:
            cell = int_to_Integer(cell)
        return old_run_cell(cell, *args, **kwargs)
    shell.run_cell = my_run_cell