import logging
import os
import re
import site
import sys
from typing import List, Optional
def running_under_virtualenv() -> bool:
    """True if we're running inside a virtual environment, False otherwise."""
    return _running_under_venv() or _running_under_legacy_virtualenv()