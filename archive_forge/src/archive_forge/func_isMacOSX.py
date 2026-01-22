import os
import sys
import warnings
from time import time as seconds
from typing import Optional
def isMacOSX(self) -> bool:
    """
        Check if current platform is macOS.

        @return: C{True} if the current platform has been detected as macOS.
        """
    return self._platform == 'darwin'