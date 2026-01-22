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
@panopticon('code', '@', None)
def sysmon_py_return_arcs(self, code: CodeType, instruction_offset: int, retval: object) -> MonitorReturn:
    """Handle sys.monitoring.events.PY_RETURN events for branch coverage."""
    frame = self.callers_frame()
    code_info = self.code_infos.get(id(code))
    if code_info is not None and code_info.file_data is not None:
        last_line = self.last_lines.get(frame)
        if last_line is not None:
            arc = (last_line, -code.co_firstlineno)
            cast(Set[TArc], code_info.file_data).add(arc)
    self.last_lines.pop(frame, None)