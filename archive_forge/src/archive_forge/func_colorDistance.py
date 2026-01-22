import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def colorDistance(col1, col2):
    """Returns a number between 0 and root(3) stating how similar
    two colours are - distance in r,g,b, space.  Only used to find
    names for things."""
    return math.sqrt((col1.red - col2.red) ** 2 + (col1.green - col2.green) ** 2 + (col1.blue - col2.blue) ** 2)