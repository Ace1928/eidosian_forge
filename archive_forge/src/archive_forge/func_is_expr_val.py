import common_z3 as CM_Z3
import ctypes
from .z3 import *
def is_expr_val(v):
    """
    EXAMPLES:

    >>> is_expr_val(Int('7'))
    False
    >>> is_expr_val(IntVal('7'))
    True
    >>> is_expr_val(Bool('y'))
    False
    >>> is_expr_val(Int('x') + 7 == Int('y'))
    False
    >>> LOnOff, (On,Off) = EnumSort("LOnOff",['On','Off'])
    >>> Block,Reset,SafetyInjection=Consts("Block Reset SafetyInjection",LOnOff)
    >>> is_expr_val(LOnOff)
    False
    >>> is_expr_val(On)
    True
    >>> is_expr_val(Block)
    False
    >>> is_expr_val(SafetyInjection)
    False
    """
    return is_const(v) and v.decl().kind() != Z3_OP_UNINTERPRETED