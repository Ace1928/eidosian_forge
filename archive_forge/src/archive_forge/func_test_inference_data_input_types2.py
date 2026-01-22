import os
import numpy as np
import pytest
from ... import from_cmdstan
from ..helpers import check_multiple_attrs
def test_inference_data_input_types2(self, paths, observed_data_paths):
    """Check input types (change, see earlier)

        posterior_predictive --> List[str], variable in posterior
        observed_data_var --> List[str], variable
        """
    for key, path in paths.items():
        if 'eight' not in key:
            continue
        inference_data = self.get_inference_data(posterior=path, posterior_predictive=['y_hat'], predictions=['y_hat'], prior=path, prior_predictive=['y_hat'], observed_data=observed_data_paths[0], observed_data_var=['y'], constant_data=observed_data_paths[0], constant_data_var=['y'], predictions_constant_data=observed_data_paths[0], predictions_constant_data_var=['y'], coords={'school': np.arange(8)}, dims={'theta': ['school'], 'y': ['school'], 'log_lik': ['school'], 'y_hat': ['school'], 'eta': ['school']}, dtypes={'theta': np.int64})
        test_dict = {'posterior': ['mu', 'tau', 'theta_tilde', 'theta'], 'posterior_predictive': ['y_hat'], 'predictions': ['y_hat'], 'prior': ['mu', 'tau', 'theta_tilde', 'theta'], 'prior_predictive': ['y_hat'], 'sample_stats': ['diverging'], 'observed_data': ['y'], 'constant_data': ['y'], 'predictions_constant_data': ['y'], 'log_likelihood': ['log_lik']}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        assert isinstance(inference_data.posterior.theta.data.flat[0], np.integer)