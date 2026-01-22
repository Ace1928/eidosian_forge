from typing import Any
from .location import SourceLocation
def is_source(source: Any) -> bool:
    """Test if the given value is a Source object.

    For internal use only.
    """
    return isinstance(source, Source)