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
def computeNodeValues(cls, graph):
    """
        Compute the value (size) of each node by summing the associated links.
        """
    for node in graph['nodes']:
        source_val = np.sum([l['value'] for l in node['sourceLinks']])
        target_val = np.sum([l['value'] for l in node['targetLinks']])
        node['value'] = max([source_val, target_val])