import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_tensorflow
from thinc.util import to_categorical
from ..util import check_input_converters, make_tempdir
@pytest.mark.skipif(not has_tensorflow, reason='needs TensorFlow')
@pytest.mark.skipif(not has_cupy_gpu, reason='needs GPU/cupy')
def test_tensorflow_wrapper_to_gpu(model, X):
    model.to_gpu(0)