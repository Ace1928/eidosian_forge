import ctypes
import sys
from typing import Any
import time
from ctypes import Structure, byref, wintypes
from typing import IO, NamedTuple, Type, cast
from pip._vendor.rich.color import ColorSystem
from pip._vendor.rich.style import Style
def move_cursor_to_column(self, column: int) -> None:
    """Move cursor to the column specified by the zero-based column index, staying on the same row

        Args:
            column (int): The zero-based column index to move the cursor to.
        """
    row, _ = self.cursor_position
    SetConsoleCursorPosition(self._handle, coords=WindowsCoordinates(row, column))