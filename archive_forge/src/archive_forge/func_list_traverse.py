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
def list_traverse(self, *args: Any, **kwargs: Any) -> IterableList[IndexObjUnion]:
    """
        :return: IterableList with the results of the traversal as produced by
            traverse()
            Tree -> IterableList[Union['Submodule', 'Tree', 'Blob']]
        """
    return super()._list_traverse(*args, **kwargs)