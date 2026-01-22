import argparse
from dataclasses import dataclass
from enum import Enum
import os.path
import tempfile
import typer
from typing import Optional
import requests
from ray.tune.experiment.config_parser import _make_parser
from ray.tune.result import DEFAULT_RESULTS_DIR
def get_file_type(config_file: str) -> SupportedFileType:
    if config_file.endswith('.py'):
        file_type = SupportedFileType.python
    elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
        file_type = SupportedFileType.yaml
    else:
        raise ValueError('Unknown file type for config file: {}. Supported extensions: .py, .yml, .yaml'.format(config_file))
    return file_type