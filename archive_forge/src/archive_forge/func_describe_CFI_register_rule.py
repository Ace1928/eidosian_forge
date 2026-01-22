from collections import defaultdict
from .constants import *
from .dwarf_expr import DWARFExprParser
from .die import DIE
from ..common.utils import preserve_stream_pos, dwarf_assert, bytes2str
from .callframe import instruction_name, CIE, FDE
def describe_CFI_register_rule(rule):
    s = _DESCR_CFI_REGISTER_RULE_TYPE[rule.type]
    if rule.type in ('OFFSET', 'VAL_OFFSET'):
        s += '%+d' % rule.arg
    elif rule.type == 'REGISTER':
        s += describe_reg_name(rule.arg)
    return s