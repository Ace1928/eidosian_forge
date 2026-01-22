from binascii import b2a_hex
from pathlib import Path
from .util import pack, unpack
from git.objects import Blob
from typing import NamedTuple, Sequence, TYPE_CHECKING, Tuple, Union, cast, List
from git.types import PathLike
@property
def ctime(self) -> Tuple[int, int]:
    """
        :return:
            Tuple(int_time_seconds_since_epoch, int_nano_seconds) of the
            file's creation time
        """
    return cast(Tuple[int, int], unpack('>LL', self.ctime_bytes))