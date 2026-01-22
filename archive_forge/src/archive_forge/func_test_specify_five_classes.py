import numpy as np
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.analysers import output_analysers
def test_specify_five_classes():
    analyser = output_analysers.ClassificationAnalyser(name='a', num_classes=5)
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(10, 5)).batch(32)
    for data in dataset:
        analyser.update(data)
    analyser.finalize()
    assert analyser.num_classes == 5