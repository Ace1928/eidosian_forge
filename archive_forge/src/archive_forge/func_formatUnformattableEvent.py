from __future__ import annotations
from datetime import datetime as DateTime
from typing import Any, Callable, Iterator, Mapping, Optional, Union, cast
from constantly import NamedConstant
from twisted.python._tzhelper import FixedOffsetTimeZone
from twisted.python.failure import Failure
from twisted.python.reflect import safe_repr
from ._flatten import aFormatter, flatFormat
from ._interfaces import LogEvent
def formatUnformattableEvent(event: LogEvent, error: BaseException) -> str:
    """
    Formats an event as text that describes the event generically and a
    formatting error.

    @param event: A logging event.
    @param error: The formatting error.

    @return: A formatted string.
    """
    try:
        return 'Unable to format event {event!r}: {error}'.format(event=event, error=error)
    except BaseException:
        failure = Failure()
        text = ', '.join((' = '.join((safe_repr(key), safe_repr(value))) for key, value in event.items()))
        return 'MESSAGE LOST: unformattable object logged: {error}\nRecoverable data: {text}\nException during formatting:\n{failure}'.format(error=safe_repr(error), failure=failure, text=text)