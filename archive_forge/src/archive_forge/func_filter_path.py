from __future__ import annotations
import collections.abc as c
import itertools
import json
import os
import datetime
import configparser
import typing as t
from . import (
from ...constants import (
from ...io import (
from ...test import (
from ...target import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...config import (
from ...data import (
from ...host_configs import (
def filter_path(path_filter: str=None) -> c.Callable[[str], bool]:
    """Return a function that filters out paths which are not a subdirectory of the given path."""

    def context_filter(path_to_filter: str) -> bool:
        """Return true if the given path matches, otherwise return False."""
        return is_subdir(path_to_filter, path_filter)
    return context_filter