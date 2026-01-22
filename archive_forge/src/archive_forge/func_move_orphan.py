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
def move_orphan(tt, orphan_id, parent_id):
    """See TreeTransformBase.new_orphan.

    This creates a new orphan in the `brz-orphans` dir at the root of the
    `TreeTransform`.

    :param tt: The TreeTransform orphaning `trans_id`.

    :param orphan_id: The trans id that should be orphaned.

    :param parent_id: The orphan parent trans id.
    """
    orphan_dir_basename = 'brz-orphans'
    od_id = tt.trans_id_tree_path(orphan_dir_basename)
    if tt.final_kind(od_id) is None:
        tt.create_directory(od_id)
    parent_path = tt._tree_id_paths[parent_id]
    actual_name = tt.final_name(orphan_id)
    new_name = tt._available_backup_name(actual_name, od_id)
    tt.adjust_path(new_name, od_id, orphan_id)
    trace.warning('%s has been orphaned in %s' % (joinpath(parent_path, actual_name), orphan_dir_basename))