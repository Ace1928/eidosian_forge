import numpy as np
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.analysers import output_analysers
def test_multi_label_two_classes_has_two_columns():
    analyser = output_analysers.ClassificationAnalyser(name='a', multi_label=True)
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(10, 2)).batch(32)
    for data in dataset:
        analyser.update(data)
    analyser.finalize()
    assert analyser.encoded