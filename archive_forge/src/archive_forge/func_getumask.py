import builtins
import os
import shutil
import sys
from typing import IO, Any, Callable, List, Optional
def getumask() -> int:
    """Get current umask value"""
    umask = os.umask(0)
    os.umask(umask)
    return umask