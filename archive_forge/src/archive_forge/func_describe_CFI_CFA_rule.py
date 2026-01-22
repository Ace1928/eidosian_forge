from collections import defaultdict
from .constants import *
from .dwarf_expr import DWARFExprParser
from .die import DIE
from ..common.utils import preserve_stream_pos, dwarf_assert, bytes2str
from .callframe import instruction_name, CIE, FDE
def describe_CFI_CFA_rule(rule):
    if rule.expr:
        return 'exp'
    else:
        return '%s%+d' % (describe_reg_name(rule.reg), rule.offset)