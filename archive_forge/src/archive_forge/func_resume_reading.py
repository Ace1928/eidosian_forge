import asyncio
from typing import Optional, cast
from .tcp_helpers import tcp_nodelay
def resume_reading(self) -> None:
    if self._reading_paused and self.transport is not None:
        try:
            self.transport.resume_reading()
        except (AttributeError, NotImplementedError, RuntimeError):
            pass
        self._reading_paused = False