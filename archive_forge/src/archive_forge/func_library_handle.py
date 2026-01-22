import gzip
import importlib
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import cloudpickle
import numpy as np
import pytest
from _pytest.outcomes import Skipped
from packaging.version import Version
from ..data import InferenceData, from_dict
def library_handle(library):
    """Import a library and return the handle."""
    if library == 'pystan':
        try:
            module = importlib.import_module('pystan')
        except ImportError:
            module = importlib.import_module('stan')
    else:
        module = importlib.import_module(library)
    return module