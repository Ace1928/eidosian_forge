import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def opaqueColor(c):
    """utility to check we have a color that's not fully transparent"""
    return isinstance(c, Color) and c.alpha > 0