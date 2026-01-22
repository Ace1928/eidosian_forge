import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def int_rgba(self):
    v = self.bitmap_rgba()
    return int((v[0] << 24 | v[1] << 16 | v[2] << 8 | v[3]) & 16777215)