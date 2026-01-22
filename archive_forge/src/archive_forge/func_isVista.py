import os
import sys
import warnings
from time import time as seconds
from typing import Optional
def isVista(self) -> bool:
    """
        Check if current platform is Windows Vista or Windows Server 2008.

        @return: C{True} if the current platform has been detected as Vista
        """
    return sys.platform == 'win32' and sys.getwindowsversion().major == 6