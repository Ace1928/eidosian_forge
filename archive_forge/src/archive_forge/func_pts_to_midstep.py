import warnings
import numpy as np
import param
from packaging.version import Version
from param import _is_number
from ..core import (
from ..core.data import ArrayInterface, DictInterface, PandasInterface, default_datatype
from ..core.data.util import dask_array_module
from ..core.util import (
from ..element.chart import Histogram, Scatter
from ..element.path import Contours, Polygons
from ..element.raster import RGB, Image
from ..element.util import categorical_aggregate2d  # noqa (API import)
from ..streams import RangeXY
from ..util.locator import MaxNLocator
@classmethod
def pts_to_midstep(cls, x, values):
    steps = np.zeros(2 * len(x))
    value_steps = tuple((np.empty(2 * len(x), dtype=v.dtype) for v in values))
    steps[1:-1:2] = steps[2::2] = x[:-1] + (x[1:] - x[:-1]) / 2
    steps[0], steps[-1] = (x[0], x[-1])
    val_arrays = []
    for v, s in zip(values, value_steps):
        s[0::2] = v
        s[1::2] = s[0::2]
        val_arrays.append(s)
    return (steps, tuple(val_arrays))