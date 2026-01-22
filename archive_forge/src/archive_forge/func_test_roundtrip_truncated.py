import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
def test_roundtrip_truncated():
    for arr in basic_arrays:
        if arr.dtype != object:
            assert_raises(ValueError, roundtrip_truncated, arr)