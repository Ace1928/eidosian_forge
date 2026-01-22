from collections import deque
from typing import Deque, Optional
from zope.interface import implementer
from ._interfaces import ILogObserver, LogEvent
def replayTo(self, otherObserver: ILogObserver) -> None:
    """
        Re-play the buffered events to another log observer.

        @param otherObserver: An observer to replay events to.
        """
    for event in self._buffer:
        otherObserver(event)