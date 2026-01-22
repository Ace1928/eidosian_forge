from __future__ import annotations
import sys
from ctypes import byref, windll
from ctypes.wintypes import DWORD, HANDLE
from typing import Any, TextIO
from prompt_toolkit.data_structures import Size
from prompt_toolkit.win32_types import STD_OUTPUT_HANDLE
from .base import Output
from .color_depth import ColorDepth
from .vt100 import Vt100_Output
from .win32 import Win32Output

        Return the default color depth for a windows terminal.

        Contrary to the Vt100 implementation, this doesn't depend on a $TERM
        variable.
        