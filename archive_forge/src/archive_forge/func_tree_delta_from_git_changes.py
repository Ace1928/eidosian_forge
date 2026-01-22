import errno
import os
import posixpath
import stat
from collections import deque
from functools import partial
from io import BytesIO
from typing import Union, List, Tuple, Set
from dulwich.config import ConfigFile as GitConfigFile
from dulwich.config import parse_submodules
from dulwich.diff_tree import RenameDetector, tree_changes
from dulwich.errors import NotTreeError
from dulwich.index import (Index, IndexEntry, blob_from_path_and_stat,
from dulwich.object_store import OverlayObjectStore, iter_tree_contents, BaseObjectStore
from dulwich.objects import S_IFGITLINK, S_ISGITLINK, ZERO_SHA, Blob, Tree, ObjectID
from .. import controldir as _mod_controldir
from .. import delta, errors, mutabletree, osutils, revisiontree, trace
from .. import transport as _mod_transport
from .. import tree as _mod_tree
from .. import urlutils, workingtree
from ..bzr.inventorytree import InventoryTreeChange
from ..revision import CURRENT_REVISION, NULL_REVISION
from ..transport import get_transport
from ..tree import MissingNestedTree, TreeEntry
from .mapping import (decode_git_path, default_mapping, encode_git_path,
def tree_delta_from_git_changes(changes, mappings, specific_files=None, require_versioned=False, include_root=False, source_extras=None, target_extras=None):
    """Create a TreeDelta from two git trees.

    source and target are iterators over tuples with:
        (filename, sha, mode)
    """
    old_mapping, new_mapping = mappings
    if target_extras is None:
        target_extras = set()
    if source_extras is None:
        source_extras = set()
    ret = delta.TreeDelta()
    added = []
    for change_type, old, new in changes:
        oldpath, oldmode, oldsha = old
        newpath, newmode, newsha = new
        if newpath == b'' and (not include_root):
            continue
        copied = change_type == 'copy'
        if oldpath is not None:
            oldpath_decoded = decode_git_path(oldpath)
        else:
            oldpath_decoded = None
        if newpath is not None:
            newpath_decoded = decode_git_path(newpath)
        else:
            newpath_decoded = None
        if not (specific_files is None or (oldpath is not None and osutils.is_inside_or_parent_of_any(specific_files, oldpath_decoded)) or (newpath is not None and osutils.is_inside_or_parent_of_any(specific_files, newpath_decoded))):
            continue
        if oldpath is None:
            oldexe = None
            oldkind = None
            oldname = None
            oldparent = None
            oldversioned = False
        else:
            oldversioned = oldpath not in source_extras
            if oldmode:
                oldexe = mode_is_executable(oldmode)
                oldkind = mode_kind(oldmode)
            else:
                oldexe = False
                oldkind = None
            if oldpath == b'':
                oldparent = None
                oldname = ''
            else:
                oldparentpath, oldname = osutils.split(oldpath_decoded)
                oldparent = old_mapping.generate_file_id(oldparentpath)
        if newpath is None:
            newexe = None
            newkind = None
            newname = None
            newparent = None
            newversioned = False
        else:
            newversioned = newpath not in target_extras
            if newmode:
                newexe = mode_is_executable(newmode)
                newkind = mode_kind(newmode)
            else:
                newexe = False
                newkind = None
            if newpath_decoded == '':
                newparent = None
                newname = ''
            else:
                newparentpath, newname = osutils.split(newpath_decoded)
                newparent = new_mapping.generate_file_id(newparentpath)
        if oldversioned and (not copied):
            fileid = old_mapping.generate_file_id(oldpath_decoded)
        elif newversioned:
            fileid = new_mapping.generate_file_id(newpath_decoded)
        else:
            fileid = None
        if old_mapping.is_special_file(oldpath):
            oldpath = None
        if new_mapping.is_special_file(newpath):
            newpath = None
        if oldpath is None and newpath is None:
            continue
        change = InventoryTreeChange(fileid, (oldpath_decoded, newpath_decoded), oldsha != newsha, (oldversioned, newversioned), (oldparent, newparent), (oldname, newname), (oldkind, newkind), (oldexe, newexe), copied=copied)
        if newpath is not None and (not newversioned) and (newkind != 'directory'):
            change.file_id = None
            ret.unversioned.append(change)
        elif change_type == 'add':
            added.append((newpath, newkind, newsha))
        elif newpath is None or newmode == 0:
            ret.removed.append(change)
        elif change_type == 'delete':
            ret.removed.append(change)
        elif change_type == 'copy':
            if stat.S_ISDIR(oldmode) and stat.S_ISDIR(newmode):
                continue
            ret.copied.append(change)
        elif change_type == 'rename':
            if stat.S_ISDIR(oldmode) and stat.S_ISDIR(newmode):
                continue
            ret.renamed.append(change)
        elif mode_kind(oldmode) != mode_kind(newmode):
            ret.kind_changed.append(change)
        elif oldsha != newsha or oldmode != newmode:
            if stat.S_ISDIR(oldmode) and stat.S_ISDIR(newmode):
                continue
            ret.modified.append(change)
        else:
            ret.unchanged.append(change)
    implicit_dirs = {b''}
    for path, kind, sha in added:
        if kind == 'directory' or path in target_extras:
            continue
        implicit_dirs.update(osutils.parent_directories(path))
    for path, kind, sha in added:
        if kind == 'directory' and path not in implicit_dirs:
            continue
        path_decoded = decode_git_path(path)
        parent_path, basename = osutils.split(path_decoded)
        parent_id = new_mapping.generate_file_id(parent_path)
        file_id = new_mapping.generate_file_id(path_decoded)
        ret.added.append(InventoryTreeChange(file_id, (None, path_decoded), True, (False, True), (None, parent_id), (None, basename), (None, kind), (None, False)))
    return ret