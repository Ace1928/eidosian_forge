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
@panopticon('code', 'line')
def sysmon_line_arcs(self, code: CodeType, line_number: int) -> MonitorReturn:
    """Handle sys.monitoring.events.LINE events for branch coverage."""
    code_info = self.code_infos[id(code)]
    ret = None
    if code_info.file_data is not None:
        frame = self.callers_frame()
        last_line = self.last_lines.get(frame)
        if last_line is not None:
            arc = (last_line, line_number)
            cast(Set[TArc], code_info.file_data).add(arc)
        self.last_lines[frame] = line_number
    return ret