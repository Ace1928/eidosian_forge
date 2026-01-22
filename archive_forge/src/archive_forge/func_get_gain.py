import functools
import math
import operator
import textwrap
import cupy
def get_gain(poles):
    return functools.reduce(operator.mul, [(1.0 - z) * (1.0 - 1.0 / z) for z in poles])