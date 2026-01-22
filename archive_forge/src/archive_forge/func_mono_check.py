import os
import re
import struct
import threading
import functools
import inspect
from tempfile import NamedTemporaryFile
import numpy as np
from numpy import testing
from numpy.testing import (
from .. import data, io
from ..data._fetchers import _fetch
from ..util import img_as_uint, img_as_float, img_as_int, img_as_ubyte
from ._warnings import expected_warnings
import pytest
def mono_check(plugin, fmt='png'):
    """Check the roundtrip behavior for images that support most types.

    All major input types should be handled.
    """
    img = img_as_ubyte(data.moon())
    r1 = roundtrip(img, plugin, fmt)
    testing.assert_allclose(img, r1)
    img2 = img > 128
    r2 = roundtrip(img2, plugin, fmt)
    testing.assert_allclose(img2, r2.astype(bool))
    img3 = img_as_float(img)
    r3 = roundtrip(img3, plugin, fmt)
    if r3.dtype.kind == 'f':
        testing.assert_allclose(img3, r3)
    else:
        testing.assert_allclose(r3, img_as_uint(img))
    img4 = img_as_int(img)
    if fmt.lower() in ('tif', 'tiff'):
        img4 -= 100
        r4 = roundtrip(img4, plugin, fmt)
        testing.assert_allclose(r4, img4)
    else:
        r4 = roundtrip(img4, plugin, fmt)
        testing.assert_allclose(r4, img_as_uint(img4))
    img5 = img_as_uint(img)
    r5 = roundtrip(img5, plugin, fmt)
    testing.assert_allclose(r5, img5)