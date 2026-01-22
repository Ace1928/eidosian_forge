import importlib
from collections import OrderedDict
import numpy as np
import pytest
from ... import from_pystan
from ...data.io_pystan import get_draws, get_draws_stan3  # pylint: disable=unused-import
from ..helpers import (  # pylint: disable=unused-import
def test_invalid_fit(self, data):
    if pystan_version() == 2:
        model = data.model
        model_data = {'J': 8, 'y': np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]), 'sigma': np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])}
        fit_test_grad = model.sampling(data=model_data, test_grad=True, check_hmc_diagnostics=False)
        with pytest.raises(AttributeError):
            _ = from_pystan(posterior=fit_test_grad)
        fit = model.sampling(data=model_data, iter=100, chains=1, check_hmc_diagnostics=False)
        del fit.sim['samples']
        with pytest.raises(AttributeError):
            _ = from_pystan(posterior=fit)