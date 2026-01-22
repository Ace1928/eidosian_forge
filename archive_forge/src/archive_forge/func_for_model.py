import importlib
import os
import re
import warnings
from collections import OrderedDict
from typing import List, Union
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...utils import CONFIG_NAME, logging
@classmethod
def for_model(cls, model_type: str, *args, **kwargs):
    if model_type in CONFIG_MAPPING:
        config_class = CONFIG_MAPPING[model_type]
        return config_class(*args, **kwargs)
    raise ValueError(f'Unrecognized model identifier: {model_type}. Should contain one of {', '.join(CONFIG_MAPPING.keys())}')