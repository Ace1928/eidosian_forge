import bisect
import re
import traceback
import warnings
from collections import defaultdict, namedtuple
import numpy as np
import param
from packaging.version import Version
from ..core import (
from ..core.ndmapping import item_check
from ..core.operation import Operation
from ..core.options import CallbackError, Cycle
from ..core.spaces import get_nested_streams
from ..core.util import (
from ..element import Points
from ..streams import LinkedStream, Params
from ..util.transform import dim
def get_directed_graph_paths(element, arrow_length):
    """
    Computes paths for a directed path which include an arrow to
    indicate the directionality of each edge.
    """
    edgepaths = element._split_edgepaths
    edges = edgepaths.split(datatype='array', dimensions=edgepaths.kdims)
    arrows = []
    for e in edges:
        sx, sy = e[0]
        ex, ey = e[1]
        rad = np.arctan2(ey - sy, ex - sx)
        xa0 = ex - np.cos(rad + np.pi / 8) * arrow_length
        ya0 = ey - np.sin(rad + np.pi / 8) * arrow_length
        xa1 = ex - np.cos(rad - np.pi / 8) * arrow_length
        ya1 = ey - np.sin(rad - np.pi / 8) * arrow_length
        arrow = np.array([(sx, sy), (ex, ey), (np.nan, np.nan), (xa0, ya0), (ex, ey), (xa1, ya1)])
        arrows.append(arrow)
    return arrows