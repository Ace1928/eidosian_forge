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
@classmethod
def revision_id_foreign_to_bzr(cls, git_rev_id):
    """Convert a git revision id handle to a Bazaar revision id."""
    from dulwich.protocol import ZERO_SHA
    if git_rev_id == ZERO_SHA:
        return NULL_REVISION
    return b'%s:%s' % (cls.revid_prefix, git_rev_id)