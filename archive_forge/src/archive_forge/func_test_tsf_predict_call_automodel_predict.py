from unittest import mock
import autokeras as ak
from autokeras import test_utils
@mock.patch('autokeras.AutoModel.fit')
@mock.patch('autokeras.AutoModel.predict')
def test_tsf_predict_call_automodel_predict(predict, fit, tmp_path):
    auto_model = ak.TimeseriesForecaster(lookback=10, directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=test_utils.TRAIN_CSV_PATH, y='survived')
    auto_model.predict(x=test_utils.TRAIN_CSV_PATH, y='survived')
    assert predict.is_called