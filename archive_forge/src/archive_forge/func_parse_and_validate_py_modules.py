import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union
from collections import OrderedDict
import yaml
def parse_and_validate_py_modules(py_modules: List[str]) -> List[str]:
    """Parses and validates a 'py_modules' option.

    This should be a list of URIs.
    """
    if not isinstance(py_modules, list):
        raise TypeError(f'`py_modules` must be a list of strings, got {type(py_modules)}.')
    for uri in py_modules:
        validate_uri(uri)
    return py_modules