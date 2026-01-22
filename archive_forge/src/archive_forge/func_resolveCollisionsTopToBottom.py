import math
from functools import cmp_to_key
from itertools import cycle
import numpy as np
import param
from ..core.data import Dataset
from ..core.dimension import Dimension
from ..core.operation import Operation
from ..core.util import get_param_values, unique_array
from .graphs import EdgePaths, Graph, Nodes
from .util import quadratic_bezier
@classmethod
def resolveCollisionsTopToBottom(cls, nodes, y, i, alpha, py):
    for node in nodes[i:]:
        dy = (y - node['y0']) * alpha
        if dy > _Y_EPS:
            node['y0'] += dy
            node['y1'] += dy
        y = node['y1'] + py