import copy
import importlib
import json
import os
import warnings
from collections import OrderedDict
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...utils import (
from .configuration_auto import AutoConfig, model_type_to_module_name, replace_list_option_in_docstrings
def getattribute_from_module(module, attr):
    if attr is None:
        return None
    if isinstance(attr, tuple):
        return tuple((getattribute_from_module(module, a) for a in attr))
    if hasattr(module, attr):
        return getattr(module, attr)
    transformers_module = importlib.import_module('transformers')
    if module != transformers_module:
        try:
            return getattribute_from_module(transformers_module, attr)
        except ValueError:
            raise ValueError(f'Could not find {attr} neither in {module} nor in {transformers_module}!')
    else:
        raise ValueError(f'Could not find {attr} in {transformers_module}!')