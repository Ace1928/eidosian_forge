from collections import namedtuple
import contextlib
from functools import cache, wraps
import inspect
from inspect import Signature, Parameter
import logging
from numbers import Number, Real
import re
import warnings
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .colors import BoundaryNorm
from .cm import ScalarMappable
from .path import Path
from .transforms import (BboxBase, Bbox, IdentityTransform, Transform, TransformedBbox,
def get_picker(self):
    """
        Return the picking behavior of the artist.

        The possible values are described in `.set_picker`.

        See Also
        --------
        set_picker, pickable, pick
        """
    return self._picker