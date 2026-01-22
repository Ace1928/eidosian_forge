from git.exc import WorkTreeRepositoryUnsupported
from git.util import LazyMixin, join_path_native, stream_copy, bin_to_hex
import gitdb.typ as dbtyp
import os.path as osp
from .util import get_object_type_by_name
from typing import Any, TYPE_CHECKING, Union
from git.types import PathLike, Commit_ish, Lit_commit_ish
@property
def hexsha(self) -> str:
    """:return: 40 byte hex version of our 20 byte binary sha"""
    return bin_to_hex(self.binsha).decode('ascii')