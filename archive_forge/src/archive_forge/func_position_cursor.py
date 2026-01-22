import sys
from typing import Optional, Tuple
from ._loop import loop_last
from .console import Console, ConsoleOptions, RenderableType, RenderResult
from .control import Control
from .segment import ControlType, Segment
from .style import StyleType
from .text import Text
def position_cursor(self) -> Control:
    """Get control codes to move cursor to beginning of live render.

        Returns:
            Control: A control instance that may be printed.
        """
    if self._shape is not None:
        _, height = self._shape
        return Control(ControlType.CARRIAGE_RETURN, (ControlType.ERASE_IN_LINE, 2), *((ControlType.CURSOR_UP, 1), (ControlType.ERASE_IN_LINE, 2)) * (height - 1))
    return Control()