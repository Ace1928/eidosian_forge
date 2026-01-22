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
def git_snapshot(self, want_unversioned=False):
    extra = set()
    os = []
    for trans_id, path in self._list_files_by_dir():
        if not self._transform.final_is_versioned(trans_id):
            if not want_unversioned:
                continue
            extra.add(path)
        o, mode = self._transform.final_git_entry(trans_id)
        if o is not None:
            self.store.add_object(o)
            os.append((encode_git_path(path), o.id, mode))
    if not os:
        return (None, extra)
    return (commit_tree(self.store, os), extra)