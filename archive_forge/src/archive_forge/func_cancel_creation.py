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
def cancel_creation(self, trans_id):
    """Cancel the creation of new file contents."""
    del self._new_contents[trans_id]
    if trans_id in self._observed_sha1s:
        del self._observed_sha1s[trans_id]
    children = self._limbo_children.get(trans_id)
    if children is not None:
        self._rename_in_limbo(children)
        del self._limbo_children[trans_id]
        del self._limbo_children_names[trans_id]
    osutils.delete_any(self._limbo_name(trans_id))