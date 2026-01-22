from __future__ import annotations
import abc
import dataclasses
import enum
import os
import typing as t
from .constants import (
from .util import (
from .data import (
from .become import (
@property
def cgroup_enum(self) -> CGroupVersion:
    """The control group version(s) required by the container. Raises an exception if the value is invalid."""
    try:
        return CGroupVersion(self.cgroup)
    except ValueError:
        raise ValueError(f'Docker completion entry "{self.name}" has an invalid value "{self.cgroup}" for the "cgroup" setting.') from None