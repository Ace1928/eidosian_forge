import warnings
from typing import Optional
from ..rcparams import rcParams
from .base import dict_to_dataset, requires
from .inference_data import WARMUP_TAG, InferenceData
from_pytree = from_dict
@requires('prior_predictive')
def prior_predictive_to_xarray(self):
    """Convert prior_predictive samples to xarray."""
    data = self.prior_predictive
    if not isinstance(data, dict):
        raise TypeError('DictConverter.prior_predictive is not a dictionary')
    prior_predictive_attrs = self._kwargs.get('prior_predictive_attrs')
    return dict_to_dataset(data, library=None, coords=self.coords, dims=self.dims, attrs=prior_predictive_attrs, index_origin=self.index_origin)