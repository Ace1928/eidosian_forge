import errno
import itertools
import logging
import os.path
import tempfile
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import (
from pip._internal.utils.misc import enum, rmtree
def set_delete(self, kind: str, value: bool) -> None:
    """Indicate whether a TempDirectory of the given kind should be
        auto-deleted.
        """
    self._should_delete[kind] = value