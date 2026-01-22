from __future__ import annotations
import os
import stat
from pathlib import Path
from string import digits
from git.exc import WorkTreeRepositoryUnsupported
from git.objects import Object
from git.refs import SymbolicReference
from git.util import hex_to_bin, bin_to_hex, cygpath
from gitdb.exc import (
import os.path as osp
from git.cmd import Git
from typing import Union, Optional, cast, TYPE_CHECKING
from git.types import Commit_ish
def short_to_long(odb: 'GitCmdObjectDB', hexsha: str) -> Optional[bytes]:
    """:return: long hexadecimal sha1 from the given less-than-40 byte hexsha
        or None if no candidate could be found.
    :param hexsha: hexsha with less than 40 byte"""
    try:
        return bin_to_hex(odb.partial_to_complete_sha_hex(hexsha))
    except BadObject:
        return None