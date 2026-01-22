from __future__ import annotations
import math
import typing as ty
from dataclasses import dataclass, replace
import numpy as np
from nibabel.casting import able_int_type
from nibabel.fileslice import strided_scalar
from nibabel.spatialimages import SpatialImage
@property
def n_coords(self) -> int:
    """Number of coordinates

        Subclasses should override with more efficient implementations.
        """
    return self.coordinates.shape[0]