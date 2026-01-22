from collections import OrderedDict, deque
from datetime import date, time, datetime
from decimal import Decimal
from fractions import Fraction
import ast
import enum
import typing
def copy_ast_without_context(x):
    if isinstance(x, ast.AST):
        kwargs = {field: copy_ast_without_context(getattr(x, field)) for field in x._fields if field != 'ctx' if hasattr(x, field)}
        return type(x)(**kwargs)
    elif isinstance(x, list):
        return list(map(copy_ast_without_context, x))
    else:
        return x