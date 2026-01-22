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
def get_revision_id(self, commit):
    if commit.encoding:
        encoding = commit.encoding.decode('ascii')
    else:
        encoding = 'utf-8'
    if commit.message is not None:
        try:
            message, metadata = self._decode_commit_message(None, commit.message, encoding)
        except UnicodeDecodeError:
            pass
        else:
            if metadata.revision_id:
                return metadata.revision_id
    return self.revision_id_foreign_to_bzr(commit.id)