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
Serialize this tree into the stream. Assumes sorted tree data.

        .. note:: We will assume our tree data to be in a sorted state. If this is not
            the case, serialization will not generate a correct tree representation as
            these are assumed to be sorted by algorithms.
        