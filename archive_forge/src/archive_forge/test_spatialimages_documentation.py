from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from .. import load as top_load
from ..imageclasses import spatial_axes_first
from ..spatialimages import HeaderDataError, SpatialHeader, SpatialImage
from ..testing import bytesio_round_trip, deprecated_to, expires, memmap_after_ufunc
from ..tmpdirs import InTemporaryDirectory
Return image, image filename, and flag for required scaling

        Subclasses can do anything to return an image, including loading a
        pre-existing image from disk.

        Returns
        -------
        img : class:`SpatialImage` instance
        fname : str
            Image filename.
        has_scaling : bool
            True if the image array has scaling to apply to the raw image array
            data, False otherwise.
        