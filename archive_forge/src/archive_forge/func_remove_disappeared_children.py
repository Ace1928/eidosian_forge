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
def remove_disappeared_children(base_bzr_tree, path, base_tree, existing_children, lookup_object):
    """Generate an inventory delta for removed children.

    :param base_bzr_tree: Base bzr tree against which to generate the
        inventory delta.
    :param path: Path to process (unicode)
    :param base_tree: Git Tree base object
    :param existing_children: Children that still exist
    :param lookup_object: Lookup a git object by its SHA1
    :return: Inventory delta, as list
    """
    if not isinstance(path, str):
        raise TypeError(path)
    ret = []
    for name, mode, hexsha in base_tree.iteritems():
        if name in existing_children:
            continue
        c_path = posixpath.join(path, decode_git_path(name))
        file_id = base_bzr_tree.path2id(c_path)
        if file_id is None:
            raise TypeError(file_id)
        ret.append((c_path, None, file_id, None))
        if stat.S_ISDIR(mode):
            ret.extend(remove_disappeared_children(base_bzr_tree, c_path, lookup_object(hexsha), [], lookup_object))
    return ret