import itertools
from typing import Dict, List
from mypy.checker import TypeChecker
from mypy.errorcodes import ErrorCode
from mypy.errors import Errors
from mypy.nodes import CallExpr, Decorator, Expression, FuncDef, MypyFile, NameExpr
from mypy.options import Options
from mypy.plugin import CheckerPluginInterface, FunctionContext, Plugin
from mypy.subtypes import is_subtype
from mypy.types import CallableType, Instance, Type, TypeVarType
def function_hook(ctx: FunctionContext) -> Type:
    try:
        return get_reducers_type(ctx)
    except AssertionError:
        return ctx.default_return_type