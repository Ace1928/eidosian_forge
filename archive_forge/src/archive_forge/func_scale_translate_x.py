import copy
import re
import numpy as np
from plotly import colors
from ...core.util import isfinite, max_range
from ..util import color_intervals, process_cmap
def scale_translate_x(x):
    return [min(x[0] * scale_x + translate_x, 1), min(x[1] * scale_x + translate_x, 1)]