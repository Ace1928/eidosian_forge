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
def computeLinkBreadths(cls, graph):
    for node in graph['nodes']:
        node['sourceLinks'].sort(key=cmp_to_key(cls.ascendingTargetBreadth))
        node['targetLinks'].sort(key=cmp_to_key(cls.ascendingSourceBreadth))
    for node in graph['nodes']:
        y0 = node['y0']
        y1 = y0
        for link in node['sourceLinks']:
            link['y0'] = y0 + link['width'] / 2
            y0 += link['width']
        for link in node['targetLinks']:
            link['y1'] = y1 + link['width'] / 2
            y1 += link['width']