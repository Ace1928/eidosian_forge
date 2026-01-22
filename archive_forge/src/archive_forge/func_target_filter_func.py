from __future__ import annotations
import collections.abc as c
import re
import typing as t
from .....executor import (
from .....provisioning import (
from . import (
from . import (
def target_filter_func(targets: set[str]) -> set[str]:
    """Filter the given targets and return the result based on the defined includes and excludes."""
    if include_targets:
        targets &= include_targets
    if exclude_targets:
        targets -= exclude_targets
    return targets