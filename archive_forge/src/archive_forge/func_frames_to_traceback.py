from types import FrameType, TracebackType
from typing import Callable, List, Optional
def frames_to_traceback(frame: Optional[FrameType], limit: int, should_prune: Optional[Callable[[str], bool]]=None) -> Optional[TracebackType]:
    ctb: Optional[TracebackType] = None
    skipped = False
    while frame is not None and limit > 0:
        if _MODIFIED_EXCEPTION_VAR_NAME in frame.f_locals:
            return TracebackType(tb_next=None, tb_frame=frame, tb_lasti=frame.f_lasti, tb_lineno=frame.f_lineno)
        if not skipped:
            if should_prune is not None and should_prune(frame.f_globals['__name__']):
                frame = frame.f_back
                continue
            skipped = True
        if should_prune is None or not should_prune(frame.f_globals['__name__']):
            ctb = TracebackType(tb_next=ctb, tb_frame=frame, tb_lasti=frame.f_lasti, tb_lineno=frame.f_lineno)
            limit -= 1
            frame = frame.f_back
            continue
        break
    return ctb