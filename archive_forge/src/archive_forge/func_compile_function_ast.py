import re
import copy
import inspect
import ast
import textwrap
def compile_function_ast(fn_ast):
    """
    Compile function AST into a code object suitable for use in eval/exec
    """
    assert isinstance(fn_ast, ast.Module)
    fndef_ast = fn_ast.body[0]
    assert isinstance(fndef_ast, ast.FunctionDef)
    return compile(fn_ast, '<%s>' % fndef_ast.name, mode='exec')