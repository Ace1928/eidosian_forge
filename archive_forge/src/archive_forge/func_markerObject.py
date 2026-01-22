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
def markerObject(self, path, trans, fill, stroke, lw, joinstyle, capstyle):
    """Return name of a marker XObject representing the given path."""
    pathops = self.pathOperations(path, trans, simplify=False)
    key = (tuple(pathops), bool(fill), bool(stroke), joinstyle, capstyle)
    result = self.markers.get(key)
    if result is None:
        name = Name('M%d' % len(self.markers))
        ob = self.reserveObject('marker %d' % len(self.markers))
        bbox = path.get_extents(trans)
        self.markers[key] = [name, ob, bbox, lw]
    else:
        if result[-1] < lw:
            result[-1] = lw
        name = result[0]
    return name