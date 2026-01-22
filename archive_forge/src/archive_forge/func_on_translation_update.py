from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Any, Optional, Pattern
import re
import time
from pyglet import clock
from pyglet import event
from pyglet.window import key
def on_translation_update(self) -> None:
    self._list.translation[:] = (-self._layout.view_x, -self._layout.view_y, 0) * 2