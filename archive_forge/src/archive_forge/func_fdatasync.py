import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def fdatasync(self):
    """Force data out to physical disk if possible."""
    self.file_handle.flush()
    try:
        fileno = self.file_handle.fileno()
    except AttributeError:
        raise errors.TransportNotPossible()
    osutils.fdatasync(fileno)