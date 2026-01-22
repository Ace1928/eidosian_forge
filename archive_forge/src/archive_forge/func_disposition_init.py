from __future__ import annotations
from typing import TYPE_CHECKING
from coverage.types import TFileDisposition
def disposition_init(cls: type[TFileDisposition], original_filename: str) -> TFileDisposition:
    """Construct and initialize a new FileDisposition object."""
    disp = cls()
    disp.original_filename = original_filename
    disp.canonical_filename = original_filename
    disp.source_filename = None
    disp.trace = False
    disp.reason = ''
    disp.file_tracer = None
    disp.has_dynamic_filename = False
    return disp