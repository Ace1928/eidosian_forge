from collections import OrderedDict, deque
from datetime import date, time, datetime
from decimal import Decimal
from fractions import Fraction
import ast
import enum
import typing
def has_ast_name(value, node):
    value_name = safe_name(value)
    if type(value_name) is not str:
        return False
    return eq_checking_types(ast_name(node), value_name)