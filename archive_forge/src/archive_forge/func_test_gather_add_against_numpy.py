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
@pytest.mark.parametrize('ops', XP_OPS)
@given(table=strategies.arrays_BI())
def test_gather_add_against_numpy(ops, table):
    table = ops.asarray(table)
    indices = ops.xp.arange(100, dtype='i').reshape(25, 4) % table.shape[0]
    ops.xp.testing.assert_allclose(ops.gather_add(table, indices), table[indices].sum(1), atol=1e-05)