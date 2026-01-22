import os
import re
import shutil
import sys
from typing import Dict, Pattern
def color_terminal() -> bool:
    if 'NO_COLOR' in os.environ:
        return False
    if sys.platform == 'win32' and colorama is not None:
        colorama.init()
        return True
    if 'FORCE_COLOR' in os.environ:
        return True
    if not hasattr(sys.stdout, 'isatty'):
        return False
    if not sys.stdout.isatty():
        return False
    if 'COLORTERM' in os.environ:
        return True
    term = os.environ.get('TERM', 'dumb').lower()
    if term in ('xterm', 'linux') or 'color' in term:
        return True
    return False