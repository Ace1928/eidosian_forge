from itertools import groupby
import numpy as np
import pandas as pd
import param
from .dimension import Dimensioned, ViewableElement, asdim
from .layout import Composable, Layout, NdLayout
from .ndmapping import NdMapping
from .overlay import CompositeOverlay, NdOverlay, Overlayable
from .spaces import GridSpace, HoloMap
from .tree import AttrTree
from .util import get_param_values
@property
def static_dimensions(self):
    """
        Return all constant dimensions.
        """
    dimensions = []
    for dim in self.kdims:
        if len(set(self.dimension_values(dim.name))) == 1:
            dimensions.append(dim)
    return dimensions