import contextlib
import errno
import os
import time
from stat import S_IEXEC, S_ISREG
from typing import Callable
from . import config as _mod_config
from . import controldir, errors, lazy_import, lock, osutils, registry, trace
from breezy import (
from breezy.i18n import gettext
from .errors import BzrError, DuplicateKey, InternalBzrError
from .filters import ContentFilterContext, filtered_output_bytes
from .mutabletree import MutableTree
from .osutils import delete_any, file_kind, pathjoin, sha_file, splitpath
from .progress import ProgressPhase
from .transport import FileExists, NoSuchFile
from .tree import InterTree, find_previous_path
def final_kind(self, trans_id):
    """Determine the final file kind, after any changes applied.

        :return: None if the file does not exist/has no contents.  (It is
            conceivable that a path would be created without the corresponding
            contents insertion command)
        """
    if trans_id in self._new_contents:
        if trans_id in self._new_reference_revision:
            return 'tree-reference'
        return self._new_contents[trans_id]
    elif trans_id in self._removed_contents:
        return None
    else:
        return self.tree_kind(trans_id)