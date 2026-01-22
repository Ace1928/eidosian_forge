import errno
import os
import posixpath
import tempfile
import time
from stat import S_IEXEC, S_ISREG
from dulwich.index import blob_from_path_and_stat, commit_tree
from dulwich.objects import Blob
from .. import annotate, conflicts, errors, multiparent, osutils
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..i18n import gettext
from ..mutabletree import MutableTree
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..tree import InterTree, TreeChange
from .mapping import (decode_git_path, encode_git_path, mode_is_executable,
from .tree import GitTree, GitTreeDirectory, GitTreeFile, GitTreeSymlink
def iter_tree_children(self, parent_id):
    """Iterate through the entry's tree children, if any"""
    try:
        path = self._tree_id_paths[parent_id]
    except KeyError:
        return
    try:
        for child in self._tree.iter_child_entries(path):
            childpath = joinpath(path, child.name)
            yield self.trans_id_tree_path(childpath)
    except _mod_transport.NoSuchFile:
        return