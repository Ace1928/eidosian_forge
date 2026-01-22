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
@pytest.mark.parametrize('op', [NumpyOps, CupyOps])
def test_ops_consistency(op):
    """Test that specific ops don't define any methods that are not on the
    Ops base class and that all ops methods define the exact same arguments."""
    attrs = [m for m in dir(op) if not m.startswith('_')]
    for attr in attrs:
        assert hasattr(Ops, attr)
        method = getattr(op, attr)
        if hasattr(method, '__call__'):
            sig = inspect.signature(method)
            params = [p for p in sig.parameters][1:]
            base_sig = inspect.signature(getattr(Ops, attr))
            base_params = [p for p in base_sig.parameters][1:]
            assert params == base_params, attr
            defaults = [p.default for p in sig.parameters.values()][1:]
            base_defaults = [p.default for p in base_sig.parameters.values()][1:]
            assert defaults == base_defaults, attr
            annots = [p.annotation for p in sig.parameters.values()][1:]
            base_annots = [p.annotation for p in base_sig.parameters.values()][1:]
            for i, (p1, p2) in enumerate(zip(annots, base_annots)):
                if p1 != inspect.Parameter.empty and p2 != inspect.Parameter.empty:
                    assert str(p1) == str(p2), attr