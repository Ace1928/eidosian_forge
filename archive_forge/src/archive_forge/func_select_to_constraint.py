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
def select_to_constraint(cls, dataset, selection):
    """
        Transform a selection dictionary to an iris Constraint.
        """
    import iris

    def get_slicer(start, end):

        def slicer(cell):
            return start <= cell.point < end
        return slicer
    constraint_kwargs = {}
    for dim, constraint in selection.items():
        if isinstance(constraint, slice):
            constraint = (constraint.start, constraint.stop)
        if isinstance(constraint, tuple):
            if constraint == (None, None):
                continue
            constraint = get_slicer(*constraint)
        dim = dataset.get_dimension(dim, strict=True)
        constraint_kwargs[dim.name] = constraint
    return iris.Constraint(**constraint_kwargs)