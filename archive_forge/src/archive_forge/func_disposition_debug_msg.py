from __future__ import annotations
from typing import TYPE_CHECKING
from coverage.types import TFileDisposition
def disposition_debug_msg(disp: TFileDisposition) -> str:
    """Make a nice debug message of what the FileDisposition is doing."""
    if disp.trace:
        msg = f'Tracing {disp.original_filename!r}'
        if disp.original_filename != disp.source_filename:
            msg += f' as {disp.source_filename!r}'
        if disp.file_tracer:
            msg += f': will be traced by {disp.file_tracer!r}'
    else:
        msg = f'Not tracing {disp.original_filename!r}: {disp.reason}'
    return msg