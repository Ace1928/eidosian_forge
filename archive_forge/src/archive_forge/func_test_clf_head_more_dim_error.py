import numpy as np
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.analysers import output_analysers
def test_clf_head_more_dim_error():
    analyser = output_analysers.ClassificationAnalyser(name='a', num_classes=9)
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 32, 32, 3)).batch(32)
    with pytest.raises(ValueError) as info:
        for data in dataset:
            analyser.update(data)
        analyser.finalize()
    assert 'Expect the target data for a to have shape' in str(info.value)