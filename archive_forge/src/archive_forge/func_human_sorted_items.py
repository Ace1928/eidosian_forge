from __future__ import annotations
import contextlib
import datetime
import errno
import hashlib
import importlib
import importlib.util
import inspect
import locale
import os
import os.path
import re
import sys
import types
from types import ModuleType
from typing import (
from coverage import env
from coverage.exceptions import CoverageException
from coverage.types import TArc
from coverage.exceptions import *   # pylint: disable=wildcard-import
def human_sorted_items(items: Iterable[SortableItem], reverse: bool=False) -> list[SortableItem]:
    """Sort (string, ...) items the way humans expect.

    The elements of `items` can be any tuple/list. They'll be sorted by the
    first element (a string), with ties broken by the remaining elements.

    Returns the sorted list of items.
    """
    return sorted(items, key=lambda item: (_human_key(item[0]), *item[1:]), reverse=reverse)