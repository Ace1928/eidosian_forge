import functools
import logging
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from io import DEFAULT_BUFFER_SIZE, BytesIO
from os import SEEK_CUR
from typing import (
from .errors import (
def deprecate_with_replacement(old_name: str, new_name: str, removed_in: str) -> None:
    """Raise an exception that a feature will be removed, but has a replacement."""
    deprecate(DEPR_MSG.format(old_name, removed_in, new_name), 4)