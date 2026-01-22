from __future__ import annotations
import os
import stat
import tarfile
import tempfile
import time
import typing as t
from .constants import (
from .config import (
from .util import (
from .data import (
from .util_common import (
def make_executable(tar_info: tarfile.TarInfo) -> t.Optional[tarfile.TarInfo]:
    """
        Make the given file executable and readable by all, and writeable by the owner.
        Existing file type bits are preserved.
        This ensures consistency of test results when using unprivileged users.
        """
    return apply_permissions(tar_info, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH | stat.S_IWUSR)