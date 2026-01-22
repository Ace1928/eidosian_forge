import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_tensorflow
from thinc.util import to_categorical
from ..util import check_input_converters, make_tempdir
@pytest.mark.skipif(not has_tensorflow, reason='needs TensorFlow')
def test_tensorflow_wrapper_keras_subclass_decorator():
    import tensorflow as tf

    class UndecoratedModel(tf.keras.Model):

        def call(self, inputs):
            return inputs
    with pytest.raises(ValueError):
        TensorFlowWrapper(UndecoratedModel())

    @keras_subclass('TestModel', X=numpy.array([0.0, 0.0]), Y=numpy.array([0.5]), input_shape=(2,))
    class TestModel(tf.keras.Model):

        def call(self, inputs):
            return inputs
    assert isinstance(TensorFlowWrapper(TestModel()), Model)