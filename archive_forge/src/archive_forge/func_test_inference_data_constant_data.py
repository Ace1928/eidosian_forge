import numpy as np
import packaging
import pytest
from ...data.io_pyro import from_pyro  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
def test_inference_data_constant_data(self):
    x1 = 10
    x2 = 12
    y1 = torch.randn(10)

    def model_constant_data(x, y1=None):
        _x = pyro.sample('x', dist.Normal(1, 3))
        pyro.sample('y1', dist.Normal(x * _x, 1), obs=y1)
    nuts_kernel = pyro.infer.NUTS(model_constant_data)
    mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=10)
    mcmc.run(x=x1, y1=y1)
    posterior = mcmc.get_samples()
    posterior_predictive = Predictive(model_constant_data, posterior)(x1)
    predictions = Predictive(model_constant_data, posterior)(x2)
    inference_data = from_pyro(mcmc, posterior_predictive=posterior_predictive, predictions=predictions, constant_data={'x1': x1}, predictions_constant_data={'x2': x2})
    test_dict = {'posterior': ['x'], 'posterior_predictive': ['y1'], 'sample_stats': ['diverging'], 'log_likelihood': ['y1'], 'predictions': ['y1'], 'observed_data': ['y1'], 'constant_data': ['x1'], 'predictions_constant_data': ['x2']}
    fails = check_multiple_attrs(test_dict, inference_data)
    assert not fails