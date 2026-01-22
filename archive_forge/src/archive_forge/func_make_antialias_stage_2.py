from __future__ import annotations
from itertools import count
import logging
from typing import TYPE_CHECKING
from toolz import unique, concat, pluck, get, memoize
from numba import literal_unroll
import numpy as np
import xarray as xr
from .antialias import AntialiasCombination
from .reductions import SpecialColumn, UsesCudaMutex, by, category_codes, summary
from .utils import (isnull, ngjit,
def make_antialias_stage_2(reds, bases):
    self_intersect = True
    for red in reds:
        if red._antialias_requires_2_stages():
            self_intersect = False
            break

    def antialias_stage_2(array_module) -> UnzippedAntialiasStage2:
        return tuple(zip(*concat((b._antialias_stage_2(self_intersect, array_module) for b in bases))))
    return (self_intersect, antialias_stage_2)