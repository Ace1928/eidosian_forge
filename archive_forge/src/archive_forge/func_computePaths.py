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
def computePaths(self, graph):
    paths = []
    for link in graph['links']:
        source, target = (link['source'], link['target'])
        x0 = source['x1']
        x1 = target['x0']
        xmid = (x0 + x1) / 2
        y0_upper = link['y0'] + link['width'] / 2
        y0_lower = link['y0'] - link['width'] / 2
        y1_upper = link['y1'] + link['width'] / 2
        y1_lower = link['y1'] - link['width'] / 2
        start = np.array([[x0, y0_upper], [x0, y0_lower]])
        bottom = quadratic_bezier((x0, y0_lower), (x1, y1_lower), (xmid, y0_lower), (xmid, y1_lower))
        mid = np.array([[x1, y1_lower], [x1, y1_upper]])
        top = quadratic_bezier((x1, y1_upper), (x0, y0_upper), (xmid, y1_upper), (xmid, y0_upper))
        spline = np.concatenate([start, bottom, mid, top])
        paths.append(spline)
    return paths