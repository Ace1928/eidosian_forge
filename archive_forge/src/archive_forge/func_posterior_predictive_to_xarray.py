import warnings
from typing import Optional
from ..rcparams import rcParams
from .base import dict_to_dataset, requires
from .inference_data import WARMUP_TAG, InferenceData
from_pytree = from_dict
@requires(['posterior_predictive', f'{WARMUP_TAG}posterior_predictive'])
def posterior_predictive_to_xarray(self):
    """Convert posterior_predictive samples to xarray."""
    data = self._init_dict('posterior_predictive')
    data_warmup = self._init_dict(f'{WARMUP_TAG}posterior_predictive')
    if not isinstance(data, dict):
        raise TypeError('DictConverter.posterior_predictive is not a dictionary')
    if not isinstance(data_warmup, dict):
        raise TypeError('DictConverter.warmup_posterior_predictive is not a dictionary')
    posterior_predictive_attrs = self._kwargs.get('posterior_predictive_attrs')
    posterior_predictive_warmup_attrs = self._kwargs.get('posterior_predictive_warmup_attrs')
    return (dict_to_dataset(data, library=None, coords=self.coords, dims=self.dims, attrs=posterior_predictive_attrs, index_origin=self.index_origin), dict_to_dataset(data_warmup, library=None, coords=self.coords, dims=self.dims, attrs=posterior_predictive_warmup_attrs, index_origin=self.index_origin))