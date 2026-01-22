import inspect
import os
import re
import sys
import tempfile
from os.path import abspath, dirname
from typing import Any, Dict, Set, Type, TYPE_CHECKING
import torch
from torch.utils._config_module import install_config_module
def is_fbcode():
    return not hasattr(torch.version, 'git_version')