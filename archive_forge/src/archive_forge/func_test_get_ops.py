import inspect
import platform
from typing import Tuple, cast
import numpy
import pytest
from hypothesis import given, settings
from hypothesis.strategies import composite, integers
from numpy.testing import assert_allclose
from packaging.version import Version
from thinc.api import (
from thinc.backends._custom_kernels import KERNELS, KERNELS_LIST, compile_mmh
from thinc.compat import has_cupy_gpu, has_torch, torch_version
from thinc.types import Floats2d
from thinc.util import torch2xp, xp2torch
from .. import strategies
from ..strategies import arrays_BI, ndarrays_of_shape
def test_get_ops():
    assert isinstance(get_ops('numpy'), NumpyOps)
    assert isinstance(get_ops('cupy'), CupyOps)
    try:
        from thinc_apple_ops import AppleOps
        assert isinstance(get_ops('cpu'), AppleOps)
    except ImportError:
        assert isinstance(get_ops('cpu'), NumpyOps)
    try:
        from thinc_bigendian_ops import BigEndianOps
        assert isinstance(get_ops('cpu'), BigEndianOps)
    except ImportError:
        assert isinstance(get_ops('cpu'), NumpyOps)
    with pytest.raises(ValueError):
        get_ops('blah')
    ops = Ops(numpy)
    assert ops.xp == numpy