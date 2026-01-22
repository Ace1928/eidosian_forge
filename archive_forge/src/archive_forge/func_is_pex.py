from __future__ import annotations
import os
import platform
import re
import sys
def is_pex() -> bool:
    """Return if streamlit running in pex.

    Pex modifies sys.path so the pex file is the first path and that's
    how we determine we're running in the pex file.
    """
    if re.match('.*pex$', sys.path[0]):
        return True
    return False