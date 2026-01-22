import warnings
from typing import Optional
from ..rcparams import rcParams
from .base import dict_to_dataset, requires
from .inference_data import WARMUP_TAG, InferenceData
from_pytree = from_dict
Convert all available data to an InferenceData object.

        Note that if groups can not be created, then the InferenceData
        will not have those groups.
        