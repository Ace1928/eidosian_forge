import inspect
import sys
import traceback
from types import FrameType, TracebackType
from typing import Union, Iterable
from stack_data import (style_with_executing_node, Options, Line, FrameInfo, LINE_GAP,
from stack_data.utils import assert_
def format_variables(self, frame_info: FrameInfo) -> Iterable[str]:
    for var in sorted(frame_info.variables, key=lambda v: v.name):
        try:
            yield (self.format_variable(var) + '\n')
        except Exception:
            pass