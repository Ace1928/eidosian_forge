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
def ie_to_hexsha(path, ie):
    try:
        return shamap[path]
    except KeyError:
        pass
    if ie.kind == 'file':
        try:
            return idmap.lookup_blob_id(ie.file_id, ie.revision)
        except KeyError:
            blob = Blob()
            blob.data = tree.get_file_text(path)
            if add_cache_entry is not None:
                add_cache_entry(blob, (ie.file_id, ie.revision), path)
            return blob.id
    elif ie.kind == 'symlink':
        try:
            return idmap.lookup_blob_id(ie.file_id, ie.revision)
        except KeyError:
            target = tree.get_symlink_target(path)
            blob = symlink_to_blob(target)
            if add_cache_entry is not None:
                add_cache_entry(blob, (ie.file_id, ie.revision), path)
            return blob.id
    elif ie.kind == 'directory':
        ret = directory_to_tree(path, ie.children.values(), ie_to_hexsha, unusual_modes, dummy_file_name, ie.parent_id is None)
        if ret is None:
            return ret
        return ret.id
    else:
        raise AssertionError