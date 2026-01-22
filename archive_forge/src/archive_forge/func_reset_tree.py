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
def reset_tree(local_tree: WorkingTree, basis_tree: Optional[Tree]=None, subpath: str='', dirty_tracker=None) -> None:
    """Reset a tree back to its basis tree.

    This will leave ignored and detritus files alone.

    Args:
      local_tree: tree to work on
      dirty_tracker: Optional dirty tracker
      subpath: Subpath to operate on
    """
    if dirty_tracker and (not dirty_tracker.is_dirty()):
        return
    if basis_tree is None:
        basis_tree = local_tree.branch.basis_tree()
    revert(local_tree, basis_tree, [subpath] if subpath else None)
    deletables: List[str] = []
    for p in local_tree.extras():
        if not is_inside(subpath, p):
            continue
        if not local_tree.is_ignored(p):
            deletables.append(local_tree.abspath(p))
    delete_items(deletables)