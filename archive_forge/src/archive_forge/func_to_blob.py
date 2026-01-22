from binascii import b2a_hex
from pathlib import Path
from .util import pack, unpack
from git.objects import Blob
from typing import NamedTuple, Sequence, TYPE_CHECKING, Tuple, Union, cast, List
from git.types import PathLike
def to_blob(self, repo: 'Repo') -> Blob:
    """:return: Blob using the information of this index entry"""
    return Blob(repo, self.binsha, self.mode, self.path)