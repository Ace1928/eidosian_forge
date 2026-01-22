import functools
import logging
import math
import pathlib
import sys
import weakref
import numpy as np
import PIL.Image
import matplotlib as mpl
from matplotlib.backend_bases import (
from matplotlib import _api, cbook, backend_tools
from matplotlib._pylab_helpers import Gcf
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
import wx
def get_wx_font(self, s, prop):
    """Return a wx font.  Cache font instances for efficiency."""
    _log.debug('%s - get_wx_font()', type(self))
    key = hash(prop)
    font = self.fontd.get(key)
    if font is not None:
        return font
    size = self.points_to_pixels(prop.get_size_in_points())
    self.fontd[key] = font = wx.Font(pointSize=round(size), family=self.fontnames.get(prop.get_name(), wx.ROMAN), style=self.fontangles[prop.get_style()], weight=self.fontweights[prop.get_weight()])
    return font