from unittest import mock
import numpy as np
import pandas as pd
import pytest
from tensorflow import nest
import autokeras as ak
from autokeras import test_utils
@mock.patch('autokeras.AutoModel.fit')
def test_structured_reg_fit_call_auto_model_fit(fit, tmp_path):
    auto_model = ak.StructuredDataRegressor(directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=pd.read_csv(test_utils.TRAIN_CSV_PATH).to_numpy().astype(np.unicode)[:100], y=test_utils.generate_data(num_instances=100, shape=(1,)))
    assert fit.is_called