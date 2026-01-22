import numpy as np
import packaging
import pytest
from ...data.io_pyro import from_pyro  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
@pytest.mark.parametrize('log_likelihood', [True, False])
def test_log_likelihood(self, log_likelihood):
    """Test behaviour when log likelihood cannot be retrieved.

        If log_likelihood=True there is a warning to say log_likelihood group is skipped,
        if log_likelihood=False there is no warning and log_likelihood is skipped.
        """
    x = torch.randn((10, 2))
    y = torch.randn(10)

    def model_constant_data(x, y=None):
        beta = pyro.sample('beta', dist.Normal(torch.ones(2), 3))
        pyro.sample('y', dist.Normal(x.matmul(beta), 1), obs=y)
    nuts_kernel = pyro.infer.NUTS(model_constant_data)
    mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=10)
    mcmc.run(x=x, y=y)
    if log_likelihood:
        with pytest.warns(UserWarning, match='Could not get vectorized trace'):
            inference_data = from_pyro(mcmc, log_likelihood=log_likelihood)
    else:
        inference_data = from_pyro(mcmc, log_likelihood=log_likelihood)
    test_dict = {'posterior': ['beta'], 'sample_stats': ['diverging'], '~log_likelihood': [''], 'observed_data': ['y']}
    fails = check_multiple_attrs(test_dict, inference_data)
    assert not fails