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
def resolve_unversioned_parent(tt, path_tree, c_type, trans_id):
    file_id = tt.inactive_file_id(trans_id)
    if path_tree and path_tree.path2id('') == file_id:
        return
    tt.version_file(trans_id, file_id=file_id)
    yield (c_type, 'Versioned directory', trans_id)