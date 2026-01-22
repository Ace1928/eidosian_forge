import json
import logging
import os
from typing import Any, Dict, Optional
import yaml
import wandb
from wandb.errors import Error
from wandb.util import load_yaml
from . import filesystem
def dict_strip_value_dict(config_dict):
    d = dict()
    for k, v in config_dict.items():
        d[k] = v['value']
    return d