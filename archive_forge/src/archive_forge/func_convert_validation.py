from __future__ import annotations
import logging # isort:skip
import os
from os.path import join
from pathlib import Path
from typing import (
import yaml
from .util.deprecation import deprecated
from .util.paths import bokehjs_path, server_path
def convert_validation(value: str | ValidationLevel) -> ValidationLevel:
    """Convert a string to a validation level

    If a validation level is passed in, it is returned as-is.

    Args:
        value (str):
            A string value to convert to a validation level

    Returns:
        string

    Raises:
        ValueError

    """
    VALID_LEVELS = {'none', 'errors', 'all'}
    lowered = value.lower()
    if lowered in VALID_LEVELS:
        return cast(ValidationLevel, lowered)
    raise ValueError(f'Cannot convert {value!r} to validation level, valid values are: {VALID_LEVELS!r}')