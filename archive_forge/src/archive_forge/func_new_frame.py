from __future__ import annotations
import struct
from typing import Any, Awaitable, Callable, NamedTuple, Optional, Sequence, Tuple
from .. import extensions, frames
from ..exceptions import PayloadTooBig, ProtocolError
from ..frames import (  # noqa: E402, F401, I001
@property
def new_frame(self) -> frames.Frame:
    return frames.Frame(self.opcode, self.data, self.fin, self.rsv1, self.rsv2, self.rsv3)