from ..core import HoloMap
from ..core.data import DataConversion, Dataset
from .annotation import *
from .chart import *
from .chart3d import *
from .geom import *
from .graphs import *
from .path import *
from .raster import *
from .sankey import *
from .stats import *
from .tabular import *
from .tiles import *
def raster(self, kdims=None, vdims=None, groupby=None, **kwargs):
    heatmap = self.heatmap(kdims, vdims, **kwargs)
    return Raster(heatmap.data, **self._element.param.values(onlychanged=True))