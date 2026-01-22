import ast
import collections
import dataclasses
import secrets
import sys
from functools import lru_cache
from importlib.util import find_spec
from typing import Dict, List, Optional, Tuple
from black.output import out
from black.report import NothingChanged
@lru_cache
def jupyter_dependencies_are_installed(*, warn: bool) -> bool:
    installed = find_spec('tokenize_rt') is not None and find_spec('IPython') is not None
    if not installed and warn:
        msg = 'Skipping .ipynb files as Jupyter dependencies are not installed.\nYou can fix this by running ``pip install "black[jupyter]"``'
        out(msg)
    return installed