import functools
import gzip
import math
import numpy as np
from .. import _api, cbook, font_manager
from matplotlib.backend_bases import (
from matplotlib.font_manager import ttfFontProperty
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
def print_rgba(self, fobj):
    width, height = self.get_width_height()
    buf = self._get_printed_image_surface().get_data()
    fobj.write(cbook._premultiplied_argb32_to_unmultiplied_rgba8888(np.asarray(buf).reshape((width, height, 4))))