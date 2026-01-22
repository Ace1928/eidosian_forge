import os
import re
import shutil
import sys
from typing import Dict, Pattern
def terminal_safe(s: str) -> str:
    """Safely encode a string for printing to the terminal."""
    return s.encode('ascii', 'backslashreplace').decode('ascii')