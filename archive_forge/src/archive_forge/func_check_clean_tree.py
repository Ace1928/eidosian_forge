import errno
import os
import shutil
from contextlib import ExitStack
from typing import List, Optional
from .clean_tree import iter_deletables
from .errors import BzrError, DependencyNotPresent
from .osutils import is_inside
from .trace import warning
from .transform import revert
from .transport import NoSuchFile
from .tree import Tree
from .workingtree import WorkingTree
def check_clean_tree(local_tree: WorkingTree, basis_tree: Optional[Tree]=None, subpath: str='') -> None:
    """Check that a tree is clean and has no pending changes or unknown files.

    Args:
      local_tree: The tree to check
      basis_tree: Tree to check against
      subpath: Subpath of the tree to check
    Raises:
      PendingChanges: When there are pending changes
    """
    with ExitStack() as es:
        if basis_tree is None:
            es.enter_context(local_tree.lock_read())
            basis_tree = local_tree.basis_tree()
        changes = local_tree.iter_changes(basis_tree, include_unchanged=False, require_versioned=False, want_unversioned=True, specific_files=[subpath])

        def relevant(p, t):
            if not p:
                return False
            if not is_inside(subpath, p):
                return False
            if t.is_ignored(p):
                return False
            try:
                if not t.has_versioned_directories() and t.kind(p) == 'directory':
                    return False
            except NoSuchFile:
                return True
            return True
        if any((change for change in changes if relevant(change.path[0], basis_tree) or relevant(change.path[1], local_tree))):
            raise WorkspaceDirty(local_tree, subpath)