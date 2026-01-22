from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
def fireSystemEvent(eventType: str) -> None:
    """
        Fire a system-wide event.

        System-wide events are things like 'startup', 'shutdown', and
        'persist'.
        """