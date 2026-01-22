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
def ord_(b: Union[int, str, bytes]) -> Union[int, bytes]:
    if isinstance(b, str):
        return ord(b)
    return b