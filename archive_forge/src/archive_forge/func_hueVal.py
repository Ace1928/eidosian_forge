import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def hueVal(self, v):
    v = v.strip()
    try:
        c = float(v)
        return (c % 360 + 360) % 360 / 360.0
    except:
        raise ValueError('bad hue argument value %r in css color %r' % (v, self.s))