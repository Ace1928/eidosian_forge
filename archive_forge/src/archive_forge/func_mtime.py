from binascii import b2a_hex
from pathlib import Path
from .util import pack, unpack
from git.objects import Blob
from typing import NamedTuple, Sequence, TYPE_CHECKING, Tuple, Union, cast, List
from git.types import PathLike
@property
def mtime(self) -> Tuple[int, int]:
    """See ctime property, but returns modification time."""
    return cast(Tuple[int, int], unpack('>LL', self.mtime_bytes))