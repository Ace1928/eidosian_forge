from typing import cast
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_mxnet
from thinc.types import Array1d, Array2d, IntsXd
from thinc.util import to_categorical
from ..util import check_input_converters, make_tempdir
@pytest.mark.skipif(not has_mxnet, reason='needs MXNet')
def test_mxnet_wrapper_built_model(model: Model[Array2d, Array2d], X: Array2d, Y: Array1d):
    assert model.predict(X) is not None
    assert model.from_bytes(model.to_bytes()) is not None