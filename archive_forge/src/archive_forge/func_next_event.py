from __future__ import annotations
import abc
import time
import typing
from .common import BaseScreen
def next_event(self) -> tuple[float, str] | None:
    """
        Return (remaining, key) where remaining is the number of seconds
        (float) until the key repeat event should be sent, or None if no
        events are pending.
        """
    if len(self.pressed) != 1 or self.multiple_pressed:
        return None
    for key, val in self.pressed.items():
        return (max(0.0, val + self.repeat_delay - time.time()), key)
    return None