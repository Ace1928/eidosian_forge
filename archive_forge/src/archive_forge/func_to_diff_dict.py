import copy
import json
import os
import warnings
from typing import Any, Dict, Optional, Union
from .. import __version__
from ..configuration_utils import PretrainedConfig
from ..utils import (
def to_diff_dict(self) -> Dict[str, Any]:
    """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
    config_dict = self.to_dict()
    default_config_dict = GenerationConfig().to_dict()
    serializable_config_dict = {}
    for key, value in config_dict.items():
        if key not in default_config_dict or key == 'transformers_version' or value != default_config_dict[key]:
            serializable_config_dict[key] = value
    self.dict_torch_dtype_to_str(serializable_config_dict)
    return serializable_config_dict