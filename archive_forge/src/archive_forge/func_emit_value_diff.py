import sys
import json
from .symbols import *
from .symbols import Symbol
def emit_value_diff(self, a, b, s):
    if s == 1.0:
        return {}
    else:
        return [a, b]