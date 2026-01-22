import gast as ast
import numpy as np
import numbers
def builtin_folding(value):
    """ Convert builtin function to ast expression. """
    if isinstance(value, (type(None), bool)):
        name = str(value)
    else:
        try:
            name = value.__name__
        except AttributeError:
            raise ToNotEval()
    return ast.Attribute(ast.Name('builtins', ast.Load(), None, None), name, ast.Load())