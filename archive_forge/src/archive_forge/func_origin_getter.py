from __future__ import annotations
import contextlib
import logging
import math
import os
import pathlib
import re
import sys
import tempfile
from functools import partial
from hashlib import md5
from importlib.metadata import version
from typing import (
from urllib.parse import urlsplit
def origin_getter(method: str, self: Any) -> Any:
    origin = getattr(self, origin_name)
    return getattr(origin, method)