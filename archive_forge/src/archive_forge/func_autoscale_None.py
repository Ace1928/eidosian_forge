import base64
from collections.abc import Sized, Sequence, Mapping
import functools
import importlib
import inspect
import io
import itertools
from numbers import Real
import re
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import matplotlib as mpl
import numpy as np
from matplotlib import _api, _cm, cbook, scale
from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS
def autoscale_None(self, A):
    in_trf_domain = np.extract(np.isfinite(self._trf.transform(A)), A)
    if in_trf_domain.size == 0:
        in_trf_domain = np.ma.masked
    return super().autoscale_None(in_trf_domain)