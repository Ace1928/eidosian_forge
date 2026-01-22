import copy
from collections.abc import Sized
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import IdentityTransform, Affine2D
from ._enums import JoinStyle, CapStyle
def get_user_transform(self):
    """Return user supplied part of marker transform."""
    if self._user_transform is not None:
        return self._user_transform.frozen()