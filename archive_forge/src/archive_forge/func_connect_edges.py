import itertools
import numpy as np
import pandas as pd
import param
from ..core import Dataset
from ..core.boundingregion import BoundingBox
from ..core.data import PandasInterface, default_datatype
from ..core.operation import Operation
from ..core.sheetcoords import Slice
from ..core.util import (
def connect_edges(graph):
    """
    Given a Graph element containing abstract edges compute edge
    segments directly connecting the source and target nodes.  This
    operation just uses internal HoloViews operations and will be a
    lot slower than the pandas equivalent.
    """
    paths = []
    for start, end in graph.array(graph.kdims):
        start_ds = graph.nodes[:, :, start]
        end_ds = graph.nodes[:, :, end]
        if not len(start_ds) or not len(end_ds):
            raise ValueError('Could not find node positions for all edges')
        start = start_ds.array(start_ds.kdims[:2])
        end = end_ds.array(end_ds.kdims[:2])
        paths.append(np.array([start[0], end[0]]))
    return paths