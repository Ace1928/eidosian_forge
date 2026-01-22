import re
import copy
import inspect
import ast
import textwrap
def function_to_ast(fn):
    """
    Get the AST representation of a function
    """
    fn_source = textwrap.dedent(inspect.getsource(fn))
    fn_ast = ast.parse(fn_source)
    return fn_ast