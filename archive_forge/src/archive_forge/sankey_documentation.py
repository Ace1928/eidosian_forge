import numpy as np
import param
from bokeh.models import Patches
from ...core.data import Dataset
from ...core.util import dimension_sanitizer, max_range
from ...util.transform import dim
from .graphs import GraphPlot
Return the extents of the Sankey box