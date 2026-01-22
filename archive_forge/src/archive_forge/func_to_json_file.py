import copy
import json
import os
import warnings
from typing import Any, Dict, Optional, Union
from .. import __version__
from ..configuration_utils import PretrainedConfig
from ..utils import (
def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool=True):
    """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `GenerationConfig()`
                is serialized to JSON file.
        """
    with open(json_file_path, 'w', encoding='utf-8') as writer:
        writer.write(self.to_json_string(use_diff=use_diff))