import os
import stat
import sys
import time
import warnings
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
from .hooks import (
from .line_ending import BlobNormalizer, TreeBlobNormalizer
from .object_store import (
from .objects import (
from .pack import generate_unpacked_objects
from .refs import (
def get_worktree_config(self) -> 'ConfigFile':
    from .config import ConfigFile
    path = os.path.join(self.commondir(), 'config.worktree')
    try:
        return ConfigFile.from_path(path)
    except FileNotFoundError:
        cf = ConfigFile()
        cf.path = path
        return cf