from __future__ import annotations
import functools
import inspect
import os
import os.path
import sys
import threading
import traceback
from dataclasses import dataclass
from types import CodeType, FrameType
from typing import (
from coverage.debug import short_filename, short_stack
from coverage.types import (
@panopticon('code', '@')
def sysmon_py_resume_arcs(self, code: CodeType, instruction_offset: int) -> MonitorReturn:
    """Handle sys.monitoring.events.PY_RESUME events for branch coverage."""
    frame = self.callers_frame()
    self.last_lines[frame] = frame.f_lineno