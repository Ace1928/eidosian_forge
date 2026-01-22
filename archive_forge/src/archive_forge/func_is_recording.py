from __future__ import annotations
from .key_processor import KeyPress
@property
def is_recording(self) -> bool:
    """Tell whether we are recording a macro."""
    return self.current_recording is not None