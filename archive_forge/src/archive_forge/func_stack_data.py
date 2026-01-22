import ast
import html
import os
import sys
from collections import defaultdict, Counter
from enum import Enum
from textwrap import dedent
from types import FrameType, CodeType, TracebackType
from typing import (
from typing import Mapping
import executing
from asttokens.util import Token
from executing import only
from pure_eval import Evaluator, is_expression_interesting
from stack_data.utils import (
@classmethod
def stack_data(cls, frame_or_tb: Union[FrameType, TracebackType], options: Optional[Options]=None, *, collapse_repeated_frames: bool=True) -> Iterator[Union['FrameInfo', RepeatedFrames]]:
    """
        An iterator of FrameInfo and RepeatedFrames objects representing
        a full traceback or stack. Similar consecutive frames are collapsed into RepeatedFrames
        objects, so always check what type of object has been yielded.

        Pass either a frame object or a traceback object,
        and optionally an Options object to configure.
        """
    stack = list(iter_stack(frame_or_tb))
    if is_frame(frame_or_tb):
        stack = stack[::-1]

    def mapper(f):
        return cls(f, options)
    if not collapse_repeated_frames:
        yield from map(mapper, stack)
        return

    def _frame_key(x):
        frame, lineno = frame_and_lineno(x)
        return (frame.f_code, lineno)
    yield from collapse_repeated(stack, mapper=mapper, collapser=RepeatedFrames, key=_frame_key)