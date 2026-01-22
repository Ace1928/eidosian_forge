import copy
from collections.abc import Sized
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import IdentityTransform, Affine2D
from ._enums import JoinStyle, CapStyle
def get_alt_transform(self):
    """
        Return the transform to be applied to the `.Path` from
        `MarkerStyle.get_alt_path()`.
        """
    if self._user_transform is None:
        return self._alt_transform.frozen()
    else:
        return (self._alt_transform + self._user_transform).frozen()