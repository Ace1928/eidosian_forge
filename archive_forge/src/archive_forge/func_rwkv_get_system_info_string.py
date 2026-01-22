import os
import sys
import ctypes
import pathlib
import platform
from typing import Optional, List, Tuple, Callable
def rwkv_get_system_info_string(self) -> str:
    """
        Returns system information string.
        """
    return self.library.rwkv_get_system_info_string().decode('utf-8')