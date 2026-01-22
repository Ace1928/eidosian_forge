import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def obj_R_G_B(c):
    """attempt to convert an object to (red,green,blue)"""
    if isinstance(c, Color):
        return (c.red, c.green, c.blue)
    elif isinstance(c, (tuple, list)):
        if len(c) == 3:
            return tuple(c)
        elif len(c) == 4:
            return toColor(c).rgb()
        else:
            raise ValueError('obj_R_G_B(%r) bad argument' % c)