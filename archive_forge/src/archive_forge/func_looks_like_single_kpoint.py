import re
import warnings
from typing import Dict
import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.cell import Cell
def looks_like_single_kpoint(obj):
    if isinstance(obj, str):
        return True
    try:
        arr = np.asarray(obj, float)
    except ValueError:
        return False
    else:
        return arr.shape == (3,)