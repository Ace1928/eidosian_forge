from typing import Any, Dict, Iterator, List, Set, no_type_check
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from qpd.constants import AGGREGATION_FUNCTIONS, WINDOW_FUNCTIONS
def make_windowframe_spec(frame_type: str, start: Any=None, end: Any=None) -> WindowFrameSpec:
    frame_type = frame_type.lower()
    if frame_type == '':
        return NoWindowFrame()
    if frame_type == 'rows':
        if isinstance(start, str):
            start = int(start)
        if isinstance(end, str):
            end = int(end)
        if start is None and end is None:
            return NoWindowFrame()
        return RollingWindowFrame(start, end)
    raise NotImplementedError(frame_type)