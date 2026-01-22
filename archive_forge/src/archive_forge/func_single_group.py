import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def single_group(vals, positions):
    result = []
    if find_min:
        result += [vals.min()]
    if find_min_positions:
        result += [positions[vals == vals.min()][0]]
    if find_max:
        result += [vals.max()]
    if find_max_positions:
        result += [positions[vals == vals.max()][0]]
    if find_median:
        result += [cupy.median(vals)]
    return result