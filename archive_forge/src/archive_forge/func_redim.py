import sys
import datetime
from itertools import product
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import Interface, DataError
from holoviews.core.data.grid import GridInterface
from holoviews.core.dimension import Dimension, asdim
from holoviews.core.element import Element
from holoviews.core.ndmapping import (NdMapping, item_check, sorted_context)
from holoviews.core.spaces import HoloMap
from holoviews.core import util
@classmethod
def redim(cls, dataset, dimensions):
    """
        Rename coords on the Cube.
        """
    new_dataset = dataset.data.copy()
    for name, new_dim in dimensions.items():
        if name == new_dataset.name():
            new_dataset.rename(new_dim.name)
        for coord in new_dataset.dim_coords:
            if name == coord.name():
                coord.rename(new_dim.name)
    return new_dataset