from __future__ import annotations
import struct
from typing import Any, Awaitable, Callable, NamedTuple, Optional, Sequence, Tuple
from .. import extensions, frames
from ..exceptions import PayloadTooBig, ProtocolError
from ..frames import (  # noqa: E402, F401, I001
def parse_close(data: bytes) -> Tuple[int, str]:
    """
    Parse the payload from a close frame.

    Returns:
        Close code and reason.

    Raises:
        ProtocolError: If data is ill-formed.
        UnicodeDecodeError: If the reason isn't valid UTF-8.

    """
    close = Close.parse(data)
    return (close.code, close.reason)