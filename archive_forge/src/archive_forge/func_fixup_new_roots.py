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
def fixup_new_roots(self):
    """Reinterpret requests to change the root directory

        Instead of creating a root directory, or moving an existing directory,
        all the attributes and children of the new root are applied to the
        existing root directory.

        This means that the old root trans-id becomes obsolete, so it is
        recommended only to invoke this after the root trans-id has become
        irrelevant.

        """
    new_roots = [k for k, v in self._new_parent.items() if v == ROOT_PARENT]
    if len(new_roots) < 1:
        return
    if len(new_roots) != 1:
        raise ValueError('A tree cannot have two roots!')
    old_new_root = new_roots[0]
    if old_new_root in self._versioned:
        self.cancel_versioning(old_new_root)
    else:
        self.unversion_file(old_new_root)
    list(self.iter_tree_children(old_new_root))
    for child in self.by_parent().get(old_new_root, []):
        self.adjust_path(self.final_name(child), self.root, child)
    if old_new_root in self._new_contents:
        self.cancel_creation(old_new_root)
    else:
        self.delete_contents(old_new_root)
    if self.root in self._removed_contents:
        self.cancel_deletion(self.root)
    del self._new_parent[old_new_root]
    del self._new_name[old_new_root]