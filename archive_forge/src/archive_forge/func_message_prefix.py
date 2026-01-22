from __future__ import annotations
import os
import os.path
import sys
from types import FrameType
from typing import Any, Iterable, Iterator
from coverage.exceptions import PluginError
from coverage.misc import isolate_module
from coverage.plugin import CoveragePlugin, FileTracer, FileReporter
from coverage.types import (
def message_prefix(self) -> str:
    """The prefix to use on messages, combining the labels."""
    prefixes = self.labels + ['']
    return ':\n'.join(('  ' * i + label for i, label in enumerate(prefixes)))