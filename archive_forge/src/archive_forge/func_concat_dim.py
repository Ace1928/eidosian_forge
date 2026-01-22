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
def concat_dim(cls, datasets, dim, vdims):
    """
        Concatenates datasets along one dimension
        """
    import iris
    try:
        from iris.util import equalise_attributes
    except ImportError:
        from iris.experimental.equalise_cubes import equalise_attributes
    cubes = []
    for c, cube in datasets.items():
        cube = cube.copy()
        cube.add_aux_coord(iris.coords.DimCoord([c], var_name=dim.name))
        cubes.append(cube)
    cubes = iris.cube.CubeList(cubes)
    equalise_attributes(cubes)
    return cubes.merge_cube()