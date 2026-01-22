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
def resolve_missing_parent(tt, path_tree, c_type, trans_id):
    if trans_id in tt._removed_contents:
        cancel_deletion = True
        orphans = tt._get_potential_orphans(trans_id)
        if orphans:
            cancel_deletion = False
            for o in orphans:
                try:
                    tt.new_orphan(o, trans_id)
                except OrphaningError:
                    cancel_deletion = True
                    break
        if cancel_deletion:
            tt.cancel_deletion(trans_id)
            yield ('deleting parent', 'Not deleting', trans_id)
    else:
        create = True
        try:
            tt.final_name(trans_id)
        except NoFinalPath:
            if path_tree is not None:
                file_id = tt.final_file_id(trans_id)
                if file_id is None:
                    file_id = tt.inactive_file_id(trans_id)
                _, entry = next(path_tree.iter_entries_by_dir(specific_files=[path_tree.id2path(file_id)]))
                if entry.parent_id is None:
                    create = False
                    moved = _reparent_transform_children(tt, trans_id, tt.root)
                    for child in moved:
                        yield (c_type, 'Moved to root', child)
                else:
                    parent_trans_id = tt.trans_id_file_id(entry.parent_id)
                    tt.adjust_path(entry.name, parent_trans_id, trans_id)
        if create:
            tt.create_directory(trans_id)
            yield (c_type, 'Created directory', trans_id)