import os
import shutil
import sys
from typing import final
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import TextIO
from .wcwidth import wcswidth
@fullwidth.setter
def fullwidth(self, value: int) -> None:
    self._terminal_width = value