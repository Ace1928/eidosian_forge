from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Any, Optional, Pattern
import re
import time
from pyglet import clock
from pyglet import event
from pyglet.window import key
def on_activate(self) -> bool:
    """Handler for the `pyglet.window.Window.on_activate` event.

        The caret is hidden when the window is not active.
        """
    self._active = True
    self.visible = self._active
    return event.EVENT_HANDLED