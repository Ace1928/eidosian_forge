import logging
import os
import re
import site
import sys
from typing import List, Optional
def virtualenv_no_global() -> bool:
    """Returns a boolean, whether running in venv with no system site-packages."""
    if _running_under_venv():
        return _no_global_under_venv()
    if _running_under_legacy_virtualenv():
        return _no_global_under_legacy_virtualenv()
    return False