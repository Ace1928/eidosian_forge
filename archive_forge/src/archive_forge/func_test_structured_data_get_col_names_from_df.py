from unittest import mock
import numpy as np
import pandas as pd
import pytest
from tensorflow import nest
import autokeras as ak
from autokeras import test_utils
@mock.patch('autokeras.AutoModel.fit')
def test_structured_data_get_col_names_from_df(fit, tmp_path):
    clf = ak.StructuredDataClassifier(directory=tmp_path, seed=test_utils.SEED)
    clf.fit(x=test_utils.TRAIN_CSV_PATH, y='survived')
    assert nest.flatten(clf.inputs)[0].column_names[0] == 'sex'