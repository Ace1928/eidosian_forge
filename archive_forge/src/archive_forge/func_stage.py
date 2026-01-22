from binascii import b2a_hex
from pathlib import Path
from .util import pack, unpack
from git.objects import Blob
from typing import NamedTuple, Sequence, TYPE_CHECKING, Tuple, Union, cast, List
from git.types import PathLike
@property
def stage(self) -> int:
    """Stage of the entry, either:

            * 0 = default stage
            * 1 = stage before a merge or common ancestor entry in case of a 3 way merge
            * 2 = stage of entries from the 'left' side of the merge
            * 3 = stage of entries from the right side of the merge

        :note: For more information, see http://www.kernel.org/pub/software/scm/git/docs/git-read-tree.html
        """
    return (self.flags & CE_STAGEMASK) >> CE_STAGESHIFT