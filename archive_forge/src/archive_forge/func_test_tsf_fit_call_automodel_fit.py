from unittest import mock
import autokeras as ak
from autokeras import test_utils
@mock.patch('autokeras.AutoModel.fit')
def test_tsf_fit_call_automodel_fit(fit, tmp_path):
    auto_model = ak.TimeseriesForecaster(lookback=10, directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=test_utils.TRAIN_CSV_PATH, y='survived', validation_data=(test_utils.TRAIN_CSV_PATH, 'survived'))
    assert fit.is_called