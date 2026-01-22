import numpy as np
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.analysers import output_analysers
def test_one_class_error():
    analyser = output_analysers.ClassificationAnalyser(name='a')
    dataset = tf.data.Dataset.from_tensor_slices(np.array(['a', 'a', 'a'])).batch(32)
    with pytest.raises(ValueError) as info:
        for data in dataset:
            analyser.update(data)
        analyser.finalize()
    assert 'Expect the target data for a to have at least 2 classes' in str(info.value)