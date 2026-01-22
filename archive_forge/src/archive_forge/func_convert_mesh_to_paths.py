import itertools
import math
from numbers import Number, Real
import warnings
import numpy as np
import matplotlib as mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
from ._enums import JoinStyle, CapStyle
@staticmethod
def convert_mesh_to_paths(tri):
    """
        Convert a given mesh into a sequence of `.Path` objects.

        This function is primarily of use to implementers of backends that do
        not directly support meshes.
        """
    triangles = tri.get_masked_triangles()
    verts = np.stack((tri.x[triangles], tri.y[triangles]), axis=-1)
    return [mpath.Path(x) for x in verts]