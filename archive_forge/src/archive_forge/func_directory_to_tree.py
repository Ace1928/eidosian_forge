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
def directory_to_tree(path, children, lookup_ie_sha1, unusual_modes, empty_file_name, allow_empty=False):
    """Create a Git Tree object from a Bazaar directory.

    :param path: directory path
    :param children: Children inventory entries
    :param lookup_ie_sha1: Lookup the Git SHA1 for a inventory entry
    :param unusual_modes: Dictionary with unusual file modes by file ids
    :param empty_file_name: Name to use for dummy files in empty directories,
        None to ignore empty directories.
    """
    tree = Tree()
    for value in children:
        if value.name in BANNED_FILENAMES:
            continue
        child_path = osutils.pathjoin(path, value.name)
        try:
            mode = unusual_modes[child_path]
        except KeyError:
            mode = entry_mode(value)
        hexsha = lookup_ie_sha1(child_path, value)
        if hexsha is not None:
            tree.add(encode_git_path(value.name), mode, hexsha)
    if not allow_empty and len(tree) == 0:
        if empty_file_name is not None:
            tree.add(empty_file_name, stat.S_IFREG | 420, Blob().id)
        else:
            return None
    return tree