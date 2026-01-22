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
def dynamic_hist(obj, **dynkwargs):
    if isinstance(obj, (NdOverlay, Overlay)):
        index = kwargs.get('index', 0)
        obj = obj.get(index)
    return obj.hist(dimension=dimension, num_bins=num_bins, bin_range=bin_range, adjoin=False, **kwargs)