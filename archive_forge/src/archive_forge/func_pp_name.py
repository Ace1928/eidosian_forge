import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def pp_name(self, a):
    r = _html_op_name(a)
    if r[0] == '&' or r[0] == '/' or r[0] == '%':
        return to_format(r, 1)
    else:
        pos = r.find('__')
        if pos == -1 or pos == 0:
            return to_format(r)
        else:
            sz = len(r)
            if pos + 2 == sz:
                return to_format(r)
            else:
                return to_format('%s<sub>%s</sub>' % (r[0:pos], r[pos + 2:sz]), sz - 2)