from __future__ import annotations
import os
from typing import TYPE_CHECKING
from monty.fnmatch import WildCard
from monty.string import list_strings
def zpath(filename: str) -> str:
    """
    Returns an existing (zipped or unzipped) file path given the unzipped
    version. If no path exists, returns the filename unmodified.

    Args:
        filename: filename without zip extension

    Returns:
        filename with a zip extension (unless an unzipped version
        exists). If filename is not found, the same filename is returned
        unchanged.
    """
    for ext in ['', '.gz', '.GZ', '.bz2', '.BZ2', '.z', '.Z']:
        zfilename = f'{filename}{ext}'
        if os.path.exists(zfilename):
            return zfilename
    return filename