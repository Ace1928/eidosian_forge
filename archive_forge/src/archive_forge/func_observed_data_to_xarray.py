import warnings
from typing import Optional
from ..rcparams import rcParams
from .base import dict_to_dataset, requires
from .inference_data import WARMUP_TAG, InferenceData
from_pytree = from_dict
@requires('observed_data')
def observed_data_to_xarray(self):
    """Convert observed_data to xarray."""
    return self.data_to_xarray(self.observed_data, group='observed_data', dims=self.dims)