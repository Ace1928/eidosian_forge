from unittest import mock
import autokeras as ak
from autokeras import test_utils
@mock.patch('autokeras.AutoModel.fit')
@mock.patch('autokeras.AutoModel.predict')
def test_tsf_predict_call_automodel_predict_fails(predict, fit, tmp_path):
    auto_model = ak.TimeseriesForecaster(lookback=10, directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=test_utils.TRAIN_CSV_PATH, y='survived')
    try:
        auto_model.predict(x=test_utils.TEST_CSV_PATH, y='survived')
    except ValueError as e:
        assert fit.is_called
        assert 'The prediction data requires the original training data to make'
        ' predictions on subsequent data points' in str(e)