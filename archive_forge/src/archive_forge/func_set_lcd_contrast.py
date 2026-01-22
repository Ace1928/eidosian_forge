from __future__ import annotations
import abc
import time
import typing
from .common import BaseScreen
def set_lcd_contrast(self, value: int) -> None:
    """
        value -- 0 to 255
        """
    if not 0 <= value <= 255:
        raise ValueError(value)
    self.queue_command(self.CMD_LCD_CONTRAST, bytearray([value]))