import numpy as np
import tensorflow as tf
from autokeras import preprocessors
from autokeras.preprocessors import postprocessors
def test_sigmoid_postprocess_to_zero_one():
    postprocessor = postprocessors.SigmoidPostprocessor()
    y = postprocessor.postprocess(np.random.rand(10, 3))
    assert set(y.flatten().tolist()) == set([1, 0])