import numpy as np
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.analysers import output_analysers
def test_reg_with_specified_output_dim_and_single_column_doesnt_crash():
    analyser = output_analysers.RegressionAnalyser(name='a', output_dim=1)
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(10)).batch(32)
    for data in dataset:
        analyser.update(data)
    analyser.finalize()