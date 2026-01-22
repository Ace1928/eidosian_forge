import copy
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.analysers import input_analysers
def test_text_analyzer_with_one_dim_doesnt_crash():
    analyser = input_analysers.TextAnalyser()
    dataset = tf.data.Dataset.from_tensor_slices(['a b c', 'b b c']).batch(32)
    for data in dataset:
        analyser.update(data)
    analyser.finalize()