import os
import sys
import locale
from configparser import ConfigParser
from itertools import chain
from pathlib import Path
from typing import MutableMapping, Mapping, Any, Dict
from xdg import BaseDirectory
from .autocomplete import AutocompleteModes
def default_editor() -> str:
    """Returns the default editor."""
    return os.environ.get('VISUAL', os.environ.get('EDITOR', 'vi'))