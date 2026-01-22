import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
def get_datafile():
    """Return default path to datafile."""
    return os.path.join(os.path.dirname(__file__), 'spacegroup.dat')