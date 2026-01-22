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
def extract_unusual_modes(rev):
    try:
        foreign_revid, mapping = mapping_registry.parse_revision_id(rev.revision_id)
    except errors.InvalidRevisionId:
        return {}
    else:
        return mapping.export_unusual_file_modes(rev)