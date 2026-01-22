import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_tensorflow
from thinc.util import to_categorical
from ..util import check_input_converters, make_tempdir
@pytest.mark.skipif(not has_tensorflow, reason='needs TensorFlow')
def test_tensorflow_wrapper_accumulate_gradients(model, X, Y, answer):
    import tensorflow as tf
    optimizer = Adam()
    gradients = []
    ops = get_current_ops()
    for i in range(3):
        guesses, backprop = model(X, is_train=True)
        guesses = ops.asarray(guesses)
        d_guesses = (guesses - Y) / guesses.shape[0]
        backprop(d_guesses)
        shim_grads = [tf.identity(var) for var in model.shims[0].gradients]
        gradients.append(shim_grads)
    model.finish_update(optimizer)
    assert model.shims[0].gradients is None
    for i in range(len(gradients)):
        if i == 0:
            continue
        found_diff = False
        curr_grads = gradients[i]
        prev_grads = gradients[i - 1]
        for curr, prev in zip(curr_grads, prev_grads):
            if (prev != curr).numpy().any():
                found_diff = True
        assert found_diff is True