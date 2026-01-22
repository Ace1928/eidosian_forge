from __future__ import annotations
import importlib.util
import os
import re
import shutil
from collections import defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING
import pytest
from numpy.typing.mypy_plugin import _EXTENDED_PRECISION_LIST
def strip_func(match: re.Match[str]) -> str:
    """`re.sub` helper function for stripping module names."""
    return match.groups()[1]