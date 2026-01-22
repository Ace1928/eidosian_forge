from __future__ import annotations
import collections
import collections.abc as c
import enum
import os
import re
import itertools
import abc
import typing as t
from .encoding import (
from .io import (
from .util import (
from .data import (
def walk_units_targets() -> c.Iterable[TestTarget]:
    """Return an iterable of units targets."""
    return walk_test_targets(path=data_context().content.unit_path, module_path=data_context().content.unit_module_path, extensions=('.py',), prefix='test_')