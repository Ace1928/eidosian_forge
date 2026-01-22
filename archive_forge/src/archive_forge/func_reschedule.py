import enum
from types import TracebackType
from typing import final, Optional, Type
from . import events
from . import exceptions
from . import tasks
def reschedule(self, when: Optional[float]) -> None:
    """Reschedule the timeout."""
    if self._state is not _State.ENTERED:
        if self._state is _State.CREATED:
            raise RuntimeError('Timeout has not been entered')
        raise RuntimeError(f'Cannot change state of {self._state.value} Timeout')
    self._when = when
    if self._timeout_handler is not None:
        self._timeout_handler.cancel()
    if when is None:
        self._timeout_handler = None
    else:
        loop = events.get_running_loop()
        if when <= loop.time():
            self._timeout_handler = loop.call_soon(self._on_timeout)
        else:
            self._timeout_handler = loop.call_at(when, self._on_timeout)