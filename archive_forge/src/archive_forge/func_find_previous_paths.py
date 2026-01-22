from typing import (TYPE_CHECKING, Dict, Iterator, List, Optional, Type, Union,
from . import errors, lock, osutils
from . import revision as _mod_revision
from . import trace
from .inter import InterObject
def find_previous_paths(from_tree: Tree, to_tree: Tree, paths: List[str], recurse: str='none') -> Dict[str, Optional[str]]:
    """Find previous tree paths.

    Args:
      from_tree: From tree
      to_tree: To tree
      paths: Iterable over paths in from_tree to search for
    Returns: Dictionary mapping from from_tree paths to paths in to_tree, or
        None if there is no equivalent path.
    """
    return InterTree.get(to_tree, from_tree).find_source_paths(paths, recurse=recurse)