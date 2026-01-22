import math
import functools
import logging
def maybeRound(v, tolerance, round=otRound):
    rounded = round(v)
    return rounded if abs(rounded - v) <= tolerance else v