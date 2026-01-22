import os
import sys
import locale
from configparser import ConfigParser
from itertools import chain
from pathlib import Path
from typing import MutableMapping, Mapping, Any, Dict
from xdg import BaseDirectory
from .autocomplete import AutocompleteModes
def default_config_path() -> Path:
    """Returns bpython's default configuration file path."""
    return get_config_home() / 'config'