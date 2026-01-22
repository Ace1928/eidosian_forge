from __future__ import annotations
import re
import warnings
from traitlets.log import get_logger
from nbformat import v3 as _v_latest
from nbformat.v3 import (
from . import versions
from .converter import convert
from .reader import reads as reader_reads
from .validator import ValidationError, validate
def writes_json(nb, **kwargs):
    """DEPRECATED, use writes"""
    warnings.warn('writes_json is deprecated since nbformat 3.0, use writes', DeprecationWarning, stacklevel=2)
    return writes(nb, **kwargs)