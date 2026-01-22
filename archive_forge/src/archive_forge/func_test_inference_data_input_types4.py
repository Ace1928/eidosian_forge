import os
import numpy as np
import pytest
from ... import from_cmdstan
from ..helpers import check_multiple_attrs
def test_inference_data_input_types4(self, paths):
    """Check input types (change, see earlier)

        coords --> one to many + one to one (non-default dim)
        dims --> one to many + one to one
        """
    paths_ = paths['no_warmup']
    for path in [paths_, paths_[0]]:
        inference_data = self.get_inference_data(posterior=path, posterior_predictive=path, prior=path, prior_predictive=path, observed_data=None, observed_data_var=None, log_likelihood=False, coords={'rand': np.arange(3)}, dims={'x': ['rand']})
        test_dict = {'posterior': ['x', 'y', 'Z'], 'prior': ['x', 'y', 'Z'], 'prior_predictive': ['x', 'y', 'Z'], 'sample_stats': ['lp'], 'sample_stats_prior': ['lp'], 'posterior_predictive': ['x', 'y', 'Z'], '~log_likelihood': ['']}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails