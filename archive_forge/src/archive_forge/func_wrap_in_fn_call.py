from lib2to3.fixer_util import (FromImport, Newline, is_import,
from lib2to3.pytree import Leaf, Node
from lib2to3.pygram import python_symbols as syms
from lib2to3.pygram import token
import re
def wrap_in_fn_call(fn_name, args, prefix=None):
    """
    Example:
    >>> wrap_in_fn_call("oldstr", (arg,))
    oldstr(arg)

    >>> wrap_in_fn_call("olddiv", (arg1, arg2))
    olddiv(arg1, arg2)

    >>> wrap_in_fn_call("olddiv", [arg1, comma, arg2, comma, arg3])
    olddiv(arg1, arg2, arg3)
    """
    assert len(args) > 0
    if len(args) == 2:
        expr1, expr2 = args
        newargs = [expr1, Comma(), expr2]
    else:
        newargs = args
    return Call(Name(fn_name), newargs, prefix=prefix)