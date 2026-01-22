import os
import sys
from typing import Optional, Tuple
def glibc_version_string_confstr() -> Optional[str]:
    """Primary implementation of glibc_version_string using os.confstr."""
    if sys.platform == 'win32':
        return None
    try:
        gnu_libc_version = os.confstr('CS_GNU_LIBC_VERSION')
        if gnu_libc_version is None:
            return None
        _, version = gnu_libc_version.split()
    except (AttributeError, OSError, ValueError):
        return None
    return version