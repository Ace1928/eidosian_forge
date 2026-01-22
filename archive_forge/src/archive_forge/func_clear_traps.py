import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def clear_traps(self):
    """Reset all traps to zero"""
    for flag in self.traps:
        self.traps[flag] = 0