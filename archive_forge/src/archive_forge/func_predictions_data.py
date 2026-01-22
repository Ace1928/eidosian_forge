import numpy as np
import packaging
import pytest
from ...data.io_pyro import from_pyro  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
@pytest.fixture(scope='class')
def predictions_data(self, data, predictions_params):
    """Generate predictions for predictions_params"""
    posterior_samples = data.obj.get_samples()
    model = data.obj.kernel.model
    predictions = Predictive(model, posterior_samples)(predictions_params['J'], torch.from_numpy(predictions_params['sigma']).float())
    return predictions