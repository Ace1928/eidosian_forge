from __future__ import annotations
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union
from mypy.nodes import ARG_POS
from mypy.nodes import CallExpr
from mypy.nodes import ClassDef
from mypy.nodes import Decorator
from mypy.nodes import Expression
from mypy.nodes import FuncDef
from mypy.nodes import MemberExpr
from mypy.nodes import NameExpr
from mypy.nodes import OverloadedFuncDef
from mypy.nodes import SymbolNode
from mypy.nodes import TypeAlias
from mypy.nodes import TypeInfo
from mypy.plugin import SemanticAnalyzerPluginInterface
from mypy.types import CallableType
from mypy.types import get_proper_type
from mypy.types import Instance
from mypy.types import UnboundType
from ... import util
def expr_to_mapped_constructor(expr: Expression) -> CallExpr:
    column_descriptor = NameExpr('__sa_Mapped')
    column_descriptor.fullname = NAMED_TYPE_SQLA_MAPPED
    member_expr = MemberExpr(column_descriptor, '_empty_constructor')
    return CallExpr(member_expr, [expr], [ARG_POS], ['arg1'])