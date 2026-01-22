from ast import parse
import codecs
import collections
import operator
import os
import re
import timeit
from .compat import importlib_metadata_get
def restore__ast(_ast):
    """Attempt to restore the required classes to the _ast module if it
    appears to be missing them
    """
    if hasattr(_ast, 'AST'):
        return
    _ast.PyCF_ONLY_AST = 2 << 9
    m = compile("def foo(): pass\nclass Bar: pass\nif False: pass\nbaz = 'mako'\n1 + 2 - 3 * 4 / 5\n6 // 7 % 8 << 9 >> 10\n11 & 12 ^ 13 | 14\n15 and 16 or 17\n-baz + (not +18) - ~17\nbaz and 'foo' or 'bar'\n(mako is baz == baz) is not baz != mako\nmako > baz < mako >= baz <= mako\nmako in baz not in mako", '<unknown>', 'exec', _ast.PyCF_ONLY_AST)
    _ast.Module = type(m)
    for cls in _ast.Module.__mro__:
        if cls.__name__ == 'mod':
            _ast.mod = cls
        elif cls.__name__ == 'AST':
            _ast.AST = cls
    _ast.FunctionDef = type(m.body[0])
    _ast.ClassDef = type(m.body[1])
    _ast.If = type(m.body[2])
    _ast.Name = type(m.body[3].targets[0])
    _ast.Store = type(m.body[3].targets[0].ctx)
    _ast.Str = type(m.body[3].value)
    _ast.Sub = type(m.body[4].value.op)
    _ast.Add = type(m.body[4].value.left.op)
    _ast.Div = type(m.body[4].value.right.op)
    _ast.Mult = type(m.body[4].value.right.left.op)
    _ast.RShift = type(m.body[5].value.op)
    _ast.LShift = type(m.body[5].value.left.op)
    _ast.Mod = type(m.body[5].value.left.left.op)
    _ast.FloorDiv = type(m.body[5].value.left.left.left.op)
    _ast.BitOr = type(m.body[6].value.op)
    _ast.BitXor = type(m.body[6].value.left.op)
    _ast.BitAnd = type(m.body[6].value.left.left.op)
    _ast.Or = type(m.body[7].value.op)
    _ast.And = type(m.body[7].value.values[0].op)
    _ast.Invert = type(m.body[8].value.right.op)
    _ast.Not = type(m.body[8].value.left.right.op)
    _ast.UAdd = type(m.body[8].value.left.right.operand.op)
    _ast.USub = type(m.body[8].value.left.left.op)
    _ast.Or = type(m.body[9].value.op)
    _ast.And = type(m.body[9].value.values[0].op)
    _ast.IsNot = type(m.body[10].value.ops[0])
    _ast.NotEq = type(m.body[10].value.ops[1])
    _ast.Is = type(m.body[10].value.left.ops[0])
    _ast.Eq = type(m.body[10].value.left.ops[1])
    _ast.Gt = type(m.body[11].value.ops[0])
    _ast.Lt = type(m.body[11].value.ops[1])
    _ast.GtE = type(m.body[11].value.ops[2])
    _ast.LtE = type(m.body[11].value.ops[3])
    _ast.In = type(m.body[12].value.ops[0])
    _ast.NotIn = type(m.body[12].value.ops[1])