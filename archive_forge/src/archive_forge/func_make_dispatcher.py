from pythran.analyses import (ImportedIds, HasReturn, IsAssigned, CFG,
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
import gast as ast
from copy import deepcopy
@staticmethod
def make_dispatcher(static_expr, func_true, func_false, imported_ids):
    dispatcher_args = [static_expr, ast.Name(func_true.name, ast.Load(), None, None), ast.Name(func_false.name, ast.Load(), None, None)]
    dispatcher = ast.Call(ast.Attribute(ast.Attribute(ast.Name('builtins', ast.Load(), None, None), 'pythran', ast.Load()), 'static_if', ast.Load()), dispatcher_args, [])
    actual_call = ast.Call(dispatcher, [ast.Name(ii, ast.Load(), None, None) for ii in imported_ids], [])
    return actual_call