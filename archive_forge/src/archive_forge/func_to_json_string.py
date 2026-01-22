import copy
import json
import os
import warnings
from typing import Any, Dict, Optional, Union
from .. import __version__
from ..configuration_utils import PretrainedConfig
from ..utils import (
def to_json_string(self, use_diff: bool=True, ignore_metadata: bool=False) -> str:
    """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `GenerationConfig()`
                is serialized to JSON string.
            ignore_metadata (`bool`, *optional*, defaults to `False`):
                Whether to ignore the metadata fields present in the instance

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
    if use_diff is True:
        config_dict = self.to_diff_dict()
    else:
        config_dict = self.to_dict()
    if ignore_metadata:
        for metadata_field in METADATA_FIELDS:
            config_dict.pop(metadata_field, None)

    def convert_keys_to_string(obj):
        if isinstance(obj, dict):
            return {str(key): convert_keys_to_string(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_keys_to_string(item) for item in obj]
        else:
            return obj
    config_dict = convert_keys_to_string(config_dict)
    return json.dumps(config_dict, indent=2, sort_keys=True) + '\n'