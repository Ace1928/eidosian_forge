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
def resolve_duplicate(tt, path_tree, c_type, last_trans_id, trans_id, name):
    final_parent = tt.final_parent(last_trans_id)
    if tt.path_changed(last_trans_id):
        existing_file, new_file = (trans_id, last_trans_id)
    else:
        existing_file, new_file = (last_trans_id, trans_id)
    if not tt._tree.has_versioned_directories() and tt.final_kind(trans_id) == 'directory' and (tt.final_kind(last_trans_id) == 'directory'):
        _reparent_transform_children(tt, existing_file, new_file)
        tt.delete_contents(existing_file)
        tt.unversion_file(existing_file)
        tt.cancel_creation(existing_file)
    else:
        new_name = tt.final_name(existing_file) + '.moved'
        tt.adjust_path(new_name, final_parent, existing_file)
        yield (c_type, 'Moved existing file to', existing_file, new_file)