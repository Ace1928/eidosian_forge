import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
def refpath(self, name):
    """Return the disk path of a ref."""
    if os.path.sep != '/':
        name = name.replace(b'/', os.fsencode(os.path.sep))
    if name == HEADREF:
        return os.path.join(self.worktree_path, name)
    else:
        return os.path.join(self.path, name)