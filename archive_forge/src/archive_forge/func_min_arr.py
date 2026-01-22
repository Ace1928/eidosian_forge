from __future__ import annotations
import numba as nb
import numpy as np
import os
@arr_operator
def min_arr(src, dst):
    return min([src, dst])