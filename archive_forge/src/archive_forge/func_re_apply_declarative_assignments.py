from __future__ import annotations
from typing import List
from typing import Optional
from typing import Union
from mypy.nodes import ARG_NAMED_OPT
from mypy.nodes import Argument
from mypy.nodes import AssignmentStmt
from mypy.nodes import CallExpr
from mypy.nodes import ClassDef
from mypy.nodes import MDEF
from mypy.nodes import MemberExpr
from mypy.nodes import NameExpr
from mypy.nodes import RefExpr
from mypy.nodes import StrExpr
from mypy.nodes import SymbolTableNode
from mypy.nodes import TempNode
from mypy.nodes import TypeInfo
from mypy.nodes import Var
from mypy.plugin import SemanticAnalyzerPluginInterface
from mypy.plugins.common import add_method_to_class
from mypy.types import AnyType
from mypy.types import get_proper_type
from mypy.types import Instance
from mypy.types import NoneTyp
from mypy.types import ProperType
from mypy.types import TypeOfAny
from mypy.types import UnboundType
from mypy.types import UnionType
from . import infer
from . import util
from .names import expr_to_mapped_constructor
from .names import NAMED_TYPE_SQLA_MAPPED
def re_apply_declarative_assignments(cls: ClassDef, api: SemanticAnalyzerPluginInterface, attributes: List[util.SQLAlchemyAttribute]) -> None:
    """For multiple class passes, re-apply our left-hand side types as mypy
    seems to reset them in place.

    """
    mapped_attr_lookup = {attr.name: attr for attr in attributes}
    update_cls_metadata = False
    for stmt in cls.defs.body:
        if isinstance(stmt, AssignmentStmt) and isinstance(stmt.lvalues[0], NameExpr) and (stmt.lvalues[0].name in mapped_attr_lookup) and isinstance(stmt.lvalues[0].node, Var):
            left_node = stmt.lvalues[0].node
            python_type_for_type = mapped_attr_lookup[stmt.lvalues[0].name].type
            left_node_proper_type = get_proper_type(left_node.type)
            if isinstance(python_type_for_type, UnboundType) and (not isinstance(left_node_proper_type, UnboundType)) and (isinstance(stmt.rvalue, CallExpr) and isinstance(stmt.rvalue.callee, MemberExpr) and isinstance(stmt.rvalue.callee.expr, NameExpr) and (stmt.rvalue.callee.expr.node is not None) and (stmt.rvalue.callee.expr.node.fullname == NAMED_TYPE_SQLA_MAPPED) and (stmt.rvalue.callee.name == '_empty_constructor') and isinstance(stmt.rvalue.args[0], CallExpr) and isinstance(stmt.rvalue.args[0].callee, RefExpr)):
                new_python_type_for_type = infer.infer_type_from_right_hand_nameexpr(api, stmt, left_node, left_node_proper_type, stmt.rvalue.args[0].callee)
                if new_python_type_for_type is not None and (not isinstance(new_python_type_for_type, UnboundType)):
                    python_type_for_type = new_python_type_for_type
                    mapped_attr_lookup[stmt.lvalues[0].name].type = python_type_for_type
                    update_cls_metadata = True
            if not isinstance(left_node.type, Instance) or left_node.type.type.fullname != NAMED_TYPE_SQLA_MAPPED:
                assert python_type_for_type is not None
                left_node.type = api.named_type(NAMED_TYPE_SQLA_MAPPED, [python_type_for_type])
    if update_cls_metadata:
        util.set_mapped_attributes(cls.info, attributes)