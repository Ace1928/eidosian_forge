import os
import sys
import warnings
from time import time as seconds
from typing import Optional
def isWinNT(self) -> bool:
    """
        Are we running in Windows NT?

        This is deprecated and always returns C{True} on win32 because
        Twisted only supports Windows NT-derived platforms at this point.

        @return: C{True} if the current platform has been detected as
            Windows NT.
        """
    warnings.warn('twisted.python.runtime.Platform.isWinNT was deprecated in Twisted 13.0. Use Platform.isWindows instead.', DeprecationWarning, stacklevel=2)
    return self.isWindows()