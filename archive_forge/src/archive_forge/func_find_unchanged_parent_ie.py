import posixpath
import stat
from typing import Dict, Iterable, Iterator, List
from dulwich.object_store import BaseObjectStore
from dulwich.objects import (ZERO_SHA, Blob, Commit, ObjectID, ShaFile, Tree,
from dulwich.pack import Pack, PackData, pack_objects_to_data
from .. import errors, lru_cache, osutils, trace, ui
from ..bzr.testament import StrictTestament3
from ..lock import LogicalLockResult
from ..revision import NULL_REVISION
from ..tree import InterTree
from .cache import from_repository as cache_from_repository
from .mapping import (default_mapping, encode_git_path, entry_mode,
from .unpeel_map import UnpeelMap
def find_unchanged_parent_ie(path, kind, other, parent_trees):
    for ptree in parent_trees:
        intertree = InterTree.get(ptree, tree)
        ppath = intertree.find_source_path(path)
        if ppath is not None:
            pkind = ptree.kind(ppath)
            if kind == 'file':
                if pkind == 'file' and ptree.get_file_sha1(ppath) == other:
                    return (ptree.path2id(ppath), ptree.get_file_revision(ppath))
            if kind == 'symlink':
                if pkind == 'symlink' and ptree.get_symlink_target(ppath) == other:
                    return (ptree.path2id(ppath), ptree.get_file_revision(ppath))
    raise KeyError