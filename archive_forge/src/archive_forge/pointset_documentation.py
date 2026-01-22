from __future__ import annotations
import math
import typing as ty
from dataclasses import dataclass, replace
import numpy as np
from nibabel.casting import able_int_type
from nibabel.fileslice import strided_scalar
from nibabel.spatialimages import SpatialImage
Retrieve the coordinates

        Parameters
        ----------
        as_homogeneous : :class:`bool`
            Return homogeneous coordinates if ``True``, or Cartesian
            coordinates if ``False``.

        name : :class:`str`
            Select a particular coordinate system if more than one may exist.
            By default, `None` is equivalent to `"world"` and corresponds to
            an RAS+ coordinate system.
        