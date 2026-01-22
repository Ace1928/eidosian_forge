import inspect
import sys
import traceback
from types import FrameType, TracebackType
from typing import Union, Iterable
from stack_data import (style_with_executing_node, Options, Line, FrameInfo, LINE_GAP,
from stack_data.utils import assert_
def print_lines(self, lines, *, file=None):
    if file is None:
        file = sys.stderr
    for line in lines:
        print(line, file=file, end='')