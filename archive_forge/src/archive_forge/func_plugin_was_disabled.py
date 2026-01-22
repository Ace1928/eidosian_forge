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
def plugin_was_disabled(self, plugin: CoveragePlugin) -> None:
    """Record that `plugin` was disabled during the run."""
    self.disabled_plugins.add(plugin._coverage_plugin_name)