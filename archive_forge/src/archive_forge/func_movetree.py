import builtins
import os
import shutil
import sys
from typing import IO, Any, Callable, List, Optional
def movetree(self, destination: str) -> None:
    """
        Recursively move the file or directory to the given `destination`
        similar to the  Unix "mv" command.

        If the `destination` is a file it may be overwritten depending on the
        :func:`os.rename` semantics.
        """
    shutil.move(self, destination)