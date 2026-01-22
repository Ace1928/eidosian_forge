import os
import numpy as np
import pytest
from ... import from_cmdstan
from ..helpers import check_multiple_attrs
def test_inference_data_input_types3(self, paths, observed_data_paths):
    """Check input types (change, see earlier)

        posterior_predictive --> str, csv file
        coords --> one to many + one to one (default dim)
        dims --> one to many
        """
    for key, path in paths.items():
        if 'eight' not in key:
            continue
        post_pred = paths['eight_schools_glob']
        inference_data = self.get_inference_data(posterior=path, posterior_predictive=post_pred, prior=path, prior_predictive=post_pred, observed_data=observed_data_paths[0], observed_data_var=['y'], log_likelihood=['log_lik', 'y_hat'], coords={'school': np.arange(8), 'log_lik_dim_0': np.arange(8), 'y_hat': np.arange(8)}, dims={'theta': ['school'], 'y': ['school'], 'y_hat': ['school'], 'eta': ['school']})
        test_dict = {'posterior': ['mu', 'tau', 'theta_tilde', 'theta'], 'sample_stats': ['diverging'], 'prior': ['mu', 'tau', 'theta_tilde', 'theta'], 'prior_predictive': ['y_hat'], 'observed_data': ['y'], 'posterior_predictive': ['y_hat'], 'log_likelihood': ['log_lik', 'y_hat']}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails