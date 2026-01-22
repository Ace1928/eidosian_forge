import base64
import stat
from typing import Optional
import fastbencode as bencode
from .. import errors, foreign, trace, urlutils
from ..foreign import ForeignRevision, ForeignVcs, VcsMappingRegistry
from ..revision import NULL_REVISION, Revision
from .errors import NoPushSupport
from .hg import extract_hg_metadata, format_hg_metadata
from .roundtrip import (CommitSupplement, extract_bzr_metadata,
def symlink_to_blob(symlink_target):
    from dulwich.objects import Blob
    blob = Blob()
    if isinstance(symlink_target, str):
        symlink_target = encode_git_path(symlink_target)
    blob.data = symlink_target
    return blob