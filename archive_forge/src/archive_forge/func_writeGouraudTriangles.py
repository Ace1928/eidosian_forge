import codecs
from datetime import timezone
from datetime import datetime
from enum import Enum
from functools import total_ordering
from io import BytesIO
import itertools
import logging
import math
import os
import string
import struct
import sys
import time
import types
import warnings
import zlib
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import _api, _text_helpers, _type1font, cbook, dviread
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.figure import Figure
from matplotlib.font_manager import get_font, fontManager as _fontManager
from matplotlib._afm import AFM
from matplotlib.ft2font import (FIXED_WIDTH, ITALIC, LOAD_NO_SCALE,
from matplotlib.transforms import Affine2D, BboxBase
from matplotlib.path import Path
from matplotlib.dates import UTC
from matplotlib import _path
from . import _backend_pdf_ps
def writeGouraudTriangles(self):
    gouraudDict = dict()
    for name, ob, points, colors in self.gouraudTriangles:
        gouraudDict[name] = ob
        shape = points.shape
        flat_points = points.reshape((shape[0] * shape[1], 2))
        colordim = colors.shape[2]
        assert colordim in (1, 4)
        flat_colors = colors.reshape((shape[0] * shape[1], colordim))
        if colordim == 4:
            colordim = 3
        points_min = np.min(flat_points, axis=0) - (1 << 8)
        points_max = np.max(flat_points, axis=0) + (1 << 8)
        factor = 4294967295 / (points_max - points_min)
        self.beginStream(ob.id, None, {'ShadingType': 4, 'BitsPerCoordinate': 32, 'BitsPerComponent': 8, 'BitsPerFlag': 8, 'ColorSpace': Name('DeviceRGB' if colordim == 3 else 'DeviceGray'), 'AntiAlias': False, 'Decode': [points_min[0], points_max[0], points_min[1], points_max[1]] + [0, 1] * colordim})
        streamarr = np.empty((shape[0] * shape[1],), dtype=[('flags', 'u1'), ('points', '>u4', (2,)), ('colors', 'u1', (colordim,))])
        streamarr['flags'] = 0
        streamarr['points'] = (flat_points - points_min) * factor
        streamarr['colors'] = flat_colors[:, :colordim] * 255.0
        self.write(streamarr.tobytes())
        self.endStream()
    self.writeObject(self.gouraudObject, gouraudDict)