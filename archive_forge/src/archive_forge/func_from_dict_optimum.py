import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from packaging import version
from ..utils import is_auto_awq_available, is_torch_available, logging
@classmethod
def from_dict_optimum(cls, config_dict):
    """
        Get compatible class with optimum gptq config dict
        """
    if 'disable_exllama' in config_dict:
        config_dict['use_exllama'] = not config_dict['disable_exllama']
        config_dict['disable_exllama'] = None
    config = cls(**config_dict)
    return config