import ast
import itertools
import types
from collections import OrderedDict, Counter, defaultdict
from types import FrameType, TracebackType
from typing import (
from asttokens import ASTText
def is_frame(frame_or_tb: Union[FrameType, TracebackType]) -> bool:
    assert_(isinstance(frame_or_tb, (types.FrameType, types.TracebackType)))
    return isinstance(frame_or_tb, (types.FrameType,))