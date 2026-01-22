import itertools
import types
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import groupby
from numbers import Number
from types import FunctionType
import numpy as np
import param
from ..streams import Params, Stream, streams_list_from_dict
from . import traversal, util
from .accessors import Opts, Redim
from .dimension import Dimension, ViewableElement
from .layout import AdjointLayout, Empty, Layout, Layoutable, NdLayout
from .ndmapping import NdMapping, UniformNdMapping, item_check
from .options import Store, StoreOptions
from .overlay import CompositeOverlay, NdOverlay, Overlay, Overlayable
@property
def unbounded(self):
    """
        Returns a list of key dimensions that are unbounded, excluding
        stream parameters. If any of these key dimensions are
        unbounded, the DynamicMap as a whole is also unbounded.
        """
    unbounded_dims = []
    stream_params = set(self._stream_parameters())
    for kdim in self.kdims:
        if str(kdim) in stream_params:
            continue
        if kdim.values:
            continue
        if None in kdim.range:
            unbounded_dims.append(str(kdim))
    return unbounded_dims