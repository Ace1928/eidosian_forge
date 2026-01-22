import warnings
from typing import Optional
from ..rcparams import rcParams
from .base import dict_to_dataset, requires
from .inference_data import WARMUP_TAG, InferenceData
from_pytree = from_dict
@requires('constant_data')
def constant_data_to_xarray(self):
    """Convert constant_data to xarray."""
    return self.data_to_xarray(self.constant_data, group='constant_data')