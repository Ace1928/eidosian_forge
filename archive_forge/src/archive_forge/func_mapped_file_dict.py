from __future__ import annotations
import functools
import os
import sys
from types import FrameType
from typing import (
from coverage import env
from coverage.config import CoverageConfig
from coverage.data import CoverageData
from coverage.debug import short_stack
from coverage.disposition import FileDisposition
from coverage.exceptions import ConfigError
from coverage.misc import human_sorted_items, isolate_module
from coverage.plugin import CoveragePlugin
from coverage.pytracer import PyTracer
from coverage.sysmon import SysMonitor
from coverage.types import (
def mapped_file_dict(self, d: Mapping[str, T]) -> dict[str, T]:
    """Return a dict like d, but with keys modified by file_mapper."""
    runtime_err = None
    for _ in range(3):
        try:
            items = list(d.items())
        except RuntimeError as ex:
            runtime_err = ex
        else:
            break
    else:
        assert isinstance(runtime_err, Exception)
        raise runtime_err
    return {self.cached_mapped_file(k): v for k, v in items if v}