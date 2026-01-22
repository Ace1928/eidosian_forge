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
def import_git_tree(texts, mapping, path, name, hexshas, base_bzr_tree, parent_id, revision_id, parent_bzr_trees, lookup_object, modes, store_updater, lookup_file_id, allow_submodules=False):
    """Import a git tree object into a bzr repository.

    :param texts: VersionedFiles object to add to
    :param path: Path in the tree (str)
    :param name: Name of the tree (str)
    :param tree: A git tree object
    :param base_bzr_tree: Base inventory against which to return inventory
        delta
    :return: Inventory delta for this subtree
    """
    base_hexsha, hexsha = hexshas
    base_mode, mode = modes
    if not isinstance(path, bytes):
        raise TypeError(path)
    if not isinstance(name, bytes):
        raise TypeError(name)
    if base_hexsha == hexsha and base_mode == mode:
        return ([], {})
    invdelta = []
    file_id = lookup_file_id(osutils.safe_unicode(path))
    ie = InventoryDirectory(file_id, decode_git_path(name), parent_id)
    tree = lookup_object(hexsha)
    if base_hexsha is None:
        base_tree = None
        old_path = None
    else:
        base_tree = lookup_object(base_hexsha)
        old_path = decode_git_path(path)
    new_path = decode_git_path(path)
    if base_tree is None or type(base_tree) is not Tree:
        ie.revision = revision_id
        invdelta.append((old_path, new_path, ie.file_id, ie))
        texts.insert_record_stream([ChunkedContentFactory((ie.file_id, ie.revision), (), None, [])])
    existing_children = set()
    child_modes = {}
    for name, child_mode, child_hexsha in tree.iteritems():
        existing_children.add(name)
        child_path = posixpath.join(path, name)
        if type(base_tree) is Tree:
            try:
                child_base_mode, child_base_hexsha = base_tree[name]
            except KeyError:
                child_base_hexsha = None
                child_base_mode = 0
        else:
            child_base_hexsha = None
            child_base_mode = 0
        if stat.S_ISDIR(child_mode):
            subinvdelta, grandchildmodes = import_git_tree(texts, mapping, child_path, name, (child_base_hexsha, child_hexsha), base_bzr_tree, file_id, revision_id, parent_bzr_trees, lookup_object, (child_base_mode, child_mode), store_updater, lookup_file_id, allow_submodules=allow_submodules)
        elif S_ISGITLINK(child_mode):
            if not allow_submodules:
                raise SubmodulesRequireSubtrees()
            subinvdelta, grandchildmodes = import_git_submodule(texts, mapping, child_path, name, (child_base_hexsha, child_hexsha), base_bzr_tree, file_id, revision_id, parent_bzr_trees, lookup_object, (child_base_mode, child_mode), store_updater, lookup_file_id)
        else:
            if not mapping.is_special_file(name):
                subinvdelta = import_git_blob(texts, mapping, child_path, name, (child_base_hexsha, child_hexsha), base_bzr_tree, file_id, revision_id, parent_bzr_trees, lookup_object, (child_base_mode, child_mode), store_updater, lookup_file_id)
            else:
                subinvdelta = []
            grandchildmodes = {}
        child_modes.update(grandchildmodes)
        invdelta.extend(subinvdelta)
        if child_mode not in (stat.S_IFDIR, DEFAULT_FILE_MODE, stat.S_IFLNK, DEFAULT_FILE_MODE | 73, S_IFGITLINK):
            child_modes[child_path] = child_mode
    if base_tree is not None and type(base_tree) is Tree:
        invdelta.extend(remove_disappeared_children(base_bzr_tree, old_path, base_tree, existing_children, lookup_object))
    store_updater.add_object(tree, (file_id, revision_id), path)
    return (invdelta, child_modes)