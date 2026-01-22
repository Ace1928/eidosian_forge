from git.exc import WorkTreeRepositoryUnsupported
from git.util import LazyMixin, join_path_native, stream_copy, bin_to_hex
import gitdb.typ as dbtyp
import os.path as osp
from .util import get_object_type_by_name
from typing import Any, TYPE_CHECKING, Union
from git.types import PathLike, Commit_ish, Lit_commit_ish
@classmethod
def new_from_sha(cls, repo: 'Repo', sha1: bytes) -> Commit_ish:
    """
        :return: new object instance of a type appropriate to represent the given
            binary sha1

        :param sha1: 20 byte binary sha1
        """
    if sha1 == cls.NULL_BIN_SHA:
        return get_object_type_by_name(b'commit')(repo, sha1)
    oinfo = repo.odb.info(sha1)
    inst = get_object_type_by_name(oinfo.type)(repo, oinfo.binsha)
    inst.size = oinfo.size
    return inst