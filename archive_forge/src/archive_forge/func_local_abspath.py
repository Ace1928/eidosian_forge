import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def local_abspath(self, relpath):
    """Return the absolute path on the local filesystem.

        This function will only be defined for Transports which have a
        physical local filesystem representation.

        :raises errors.NotLocalUrl: When no local path representation is
            available.
        """
    raise errors.NotLocalUrl(self.abspath(relpath))