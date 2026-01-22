import os
import sys
import warnings
from time import time as seconds
from typing import Optional
def supportsINotify(self) -> bool:
    """
        Return C{True} if we can use the inotify API on this platform.

        @since: 10.1
        """
    try:
        from twisted.python._inotify import INotifyError, init
    except ImportError:
        return False
    try:
        os.close(init())
    except INotifyError:
        return False
    return True