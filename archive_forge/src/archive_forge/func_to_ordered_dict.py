import collections
from collections import OrderedDict
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy, copy
import itertools
from functools import reduce
from _plotly_utils.utils import (
from _plotly_utils.exceptions import PlotlyKeyError
from .optional_imports import get_module
from . import shapeannotation
from . import _subplots
def to_ordered_dict(self, skip_uid=True):
    result = collections.OrderedDict()
    result['data'] = BaseFigure._to_ordered_dict(self._data, skip_uid=skip_uid)
    result['layout'] = BaseFigure._to_ordered_dict(self._layout)
    if self._frame_objs:
        frames_props = [frame._props for frame in self._frame_objs]
        result['frames'] = BaseFigure._to_ordered_dict(frames_props)
    return result