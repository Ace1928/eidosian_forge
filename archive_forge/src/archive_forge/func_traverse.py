from git.util import IterableList, join_path
import git.diff as git_diff
from git.util import to_bin_sha
from . import util
from .base import IndexObject, IndexObjUnion
from .blob import Blob
from .submodule.base import Submodule
from .fun import tree_entries_from_data, tree_to_stream
from typing import (
from git.types import PathLike, Literal
def traverse(self, predicate: Callable[[Union[IndexObjUnion, TraversedTreeTup], int], bool]=lambda i, d: True, prune: Callable[[Union[IndexObjUnion, TraversedTreeTup], int], bool]=lambda i, d: False, depth: int=-1, branch_first: bool=True, visit_once: bool=False, ignore_self: int=1, as_edge: bool=False) -> Union[Iterator[IndexObjUnion], Iterator[TraversedTreeTup]]:
    """For documentation, see util.Traversable._traverse().

        Trees are set to ``visit_once = False`` to gain more performance in the traversal.
        """
    return cast(Union[Iterator[IndexObjUnion], Iterator[TraversedTreeTup]], super()._traverse(predicate, prune, depth, branch_first, visit_once, ignore_self))