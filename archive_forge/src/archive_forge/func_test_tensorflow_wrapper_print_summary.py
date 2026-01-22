import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_tensorflow
from thinc.util import to_categorical
from ..util import check_input_converters, make_tempdir
@pytest.mark.skipif(not has_tensorflow, reason='needs TensorFlow')
def test_tensorflow_wrapper_print_summary(model, X):
    summary = str(model.shims[0])
    assert 'layer_normalization' in summary
    assert 'dense' in summary
    assert 'Total params' in summary
    assert 'Trainable params' in summary
    assert 'Non-trainable params' in summary