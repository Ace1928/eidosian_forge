import importlib.util
import logging
import sys
import types
from typing import Any, Dict, List, Optional, Sequence, cast
import numpy as np
from ._typing import _T
def is_cudf_available() -> bool:
    """Check cuDF package available or not"""
    if importlib.util.find_spec('cudf') is None:
        return False
    try:
        import cudf
        return True
    except ImportError:
        _logger.exception('Importing cuDF failed, use DMatrix instead of QDM')
        return False