import posixpath
import stat
from dulwich.object_store import tree_lookup_path
from dulwich.objects import (S_IFGITLINK, S_ISGITLINK, ZERO_SHA, Commit, Tag,
from .. import debug, errors, osutils, trace
from ..bzr.inventory import (InventoryDirectory, InventoryFile, InventoryLink,
from ..bzr.inventorytree import InventoryRevisionTree
from ..bzr.testament import StrictTestament3
from ..bzr.versionedfile import ChunkedContentFactory
from ..errors import BzrError
from ..revision import NULL_REVISION
from ..transport import NoSuchFile
from ..tree import InterTree
from ..tsort import topo_sort
from .mapping import (DEFAULT_FILE_MODE, decode_git_path, mode_is_executable,
from .object_store import LRUTreeCache, _tree_to_objects
def verify_commit_reconstruction(target_git_object_retriever, lookup_object, o, rev, ret_tree, parent_trees, mapping, unusual_modes, verifiers):
    new_unusual_modes = mapping.export_unusual_file_modes(rev)
    if new_unusual_modes != unusual_modes:
        raise AssertionError("unusual modes don't match: {!r} != {!r}".format(unusual_modes, new_unusual_modes))
    rec_o = target_git_object_retriever._reconstruct_commit(rev, o.tree, True, verifiers)
    if rec_o != o:
        raise AssertionError('Reconstructed commit differs: {!r} != {!r}'.format(rec_o, o))
    diff = []
    new_objs = {}
    for path, obj, ie in _tree_to_objects(ret_tree, parent_trees, target_git_object_retriever._cache.idmap, unusual_modes, mapping.BZR_DUMMY_FILE):
        old_obj_id = tree_lookup_path(lookup_object, o.tree, path)[1]
        new_objs[path] = obj
        if obj.id != old_obj_id:
            diff.append((path, lookup_object(old_obj_id), obj))
    for path, old_obj, new_obj in diff:
        while old_obj.type_name == 'tree' and new_obj.type_name == 'tree' and (sorted(old_obj) == sorted(new_obj)):
            for name in old_obj:
                if old_obj[name][0] != new_obj[name][0]:
                    raise AssertionError('Modes for %s differ: %o != %o' % (path, old_obj[name][0], new_obj[name][0]))
                if old_obj[name][1] != new_obj[name][1]:
                    path = posixpath.join(path, name)
                    old_obj = lookup_object(old_obj[name][1])
                    new_obj = new_objs[path]
                    break
        raise AssertionError('objects differ for {}: {!r} != {!r}'.format(path, old_obj, new_obj))