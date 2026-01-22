import warnings
from typing import Optional
from ..rcparams import rcParams
from .base import dict_to_dataset, requires
from .inference_data import WARMUP_TAG, InferenceData
from_pytree = from_dict
@requires('predictions_constant_data')
def predictions_constant_data_to_xarray(self):
    """Convert predictions_constant_data to xarray."""
    return self.data_to_xarray(self.predictions_constant_data, group='predictions_constant_data', dims=self.pred_dims)