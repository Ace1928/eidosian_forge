from __future__ import annotations
import logging # isort:skip
import os
from os.path import join
from pathlib import Path
from typing import (
import yaml
from .util.deprecation import deprecated
from .util.paths import bokehjs_path, server_path
def convert_int(value: int | str) -> int:
    """ Convert a string to an integer.
    """
    return int(value)