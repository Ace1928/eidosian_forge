import warnings
from typing import Optional
from ..rcparams import rcParams
from .base import dict_to_dataset, requires
from .inference_data import WARMUP_TAG, InferenceData
from_pytree = from_dict
@requires('sample_stats_prior')
def sample_stats_prior_to_xarray(self):
    """Convert sample_stats_prior samples to xarray."""
    data = self.sample_stats_prior
    if not isinstance(data, dict):
        raise TypeError('DictConverter.sample_stats_prior is not a dictionary')
    sample_stats_prior_attrs = self._kwargs.get('sample_stats_prior_attrs')
    return dict_to_dataset(data, library=None, coords=self.coords, dims=self.dims, attrs=sample_stats_prior_attrs, index_origin=self.index_origin)