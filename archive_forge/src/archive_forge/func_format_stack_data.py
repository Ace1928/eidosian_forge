import inspect
import sys
import traceback
from types import FrameType, TracebackType
from typing import Union, Iterable
from stack_data import (style_with_executing_node, Options, Line, FrameInfo, LINE_GAP,
from stack_data.utils import assert_
def format_stack_data(self, stack: Iterable[Union[FrameInfo, RepeatedFrames]]) -> Iterable[str]:
    for item in stack:
        if isinstance(item, FrameInfo):
            yield from self.format_frame(item)
        else:
            yield self.format_repeated_frames(item)