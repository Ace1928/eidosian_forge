import copy
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.analysers import input_analysers
def test_structured_data_infer_col_types():
    analyser = input_analysers.StructuredDataAnalyser(column_names=test_utils.COLUMN_NAMES, column_types=None)
    x = pd.read_csv(test_utils.TRAIN_CSV_PATH)
    x.pop('survived')
    dataset = tf.data.Dataset.from_tensor_slices(x.values.astype(np.unicode)).batch(32)
    for data in dataset:
        analyser.update(data)
    analyser.finalize()
    assert analyser.column_types == test_utils.COLUMN_TYPES