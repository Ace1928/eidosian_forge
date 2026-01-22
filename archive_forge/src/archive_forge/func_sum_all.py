import collections
from functools import reduce, singledispatch
from typing import (Any, Dict, Iterable, List, Optional,
import numpy as np
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData, Info
from ase.utils.plotting import SimplePlottingAxes
def sum_all(self) -> DOSData:
    """Sum all the DOSData contained in this Collection"""
    if len(self) == 0:
        raise IndexError('No data to sum')
    elif len(self) == 1:
        data = self[0].copy()
    else:
        data = reduce(lambda x, y: x + y, self)
    return data