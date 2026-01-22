import copy
import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from packaging import version
from . import __version__
from .dynamic_module_utils import custom_object_save
from .utils import (
@classmethod
def get_config_dict(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        [`PretrainedConfig`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.

        """
    cls._set_token_in_kwargs(kwargs)
    original_kwargs = copy.deepcopy(kwargs)
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
    if '_commit_hash' in config_dict:
        original_kwargs['_commit_hash'] = config_dict['_commit_hash']
    if 'configuration_files' in config_dict:
        configuration_file = get_configuration_file(config_dict['configuration_files'])
        config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, _configuration_file=configuration_file, **original_kwargs)
    return (config_dict, kwargs)