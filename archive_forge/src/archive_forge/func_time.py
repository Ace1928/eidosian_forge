from mmap import mmap
import re
import time as _time
from git.compat import defenc
from git.objects.util import (
from git.util import (
import os.path as osp
from typing import Iterator, List, Tuple, Union, TYPE_CHECKING
from git.types import PathLike
@property
def time(self) -> Tuple[int, int]:
    """time as tuple:

        * [0] = ``int(time)``
        * [1] = ``int(timezone_offset)`` in :attr:`time.altzone` format
        """
    return self[3]