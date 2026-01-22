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
def walk_powershell_targets(include_symlinks: bool=True) -> c.Iterable[TestTarget]:
    """Return an iterable of PowerShell targets."""
    return walk_test_targets(module_path=data_context().content.module_path, extensions=('.ps1', '.psm1'), include_symlinks=include_symlinks)