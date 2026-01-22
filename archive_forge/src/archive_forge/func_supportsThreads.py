import os
import sys
import warnings
from time import time as seconds
from typing import Optional
def supportsThreads(self) -> bool:
    """
        Can threads be created?

        @return: C{True} if the threads are supported on the current platform.
        """
    try:
        import threading
        return threading is not None
    except ImportError:
        return False