import os
import shutil
import sys
from typing import final
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import TextIO
from .wcwidth import wcswidth
@property
def width_of_current_line(self) -> int:
    """Return an estimate of the width so far in the current line."""
    return wcswidth(self._current_line)