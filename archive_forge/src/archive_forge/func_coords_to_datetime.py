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
def coords_to_datetime(coords):
    nan_mask = np.isnan(coords)
    any_nan = np.any(nan_mask)
    if any_nan:
        coords[nan_mask] = 0
    coords = np.array(num2date(coords))
    if any_nan:
        coords[nan_mask] = np.nan
    return coords