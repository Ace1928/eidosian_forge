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
def export_unusual_file_modes(self, rev):
    try:
        file_modes = rev.properties['file-modes']
    except KeyError:
        return {}
    else:
        return dict(bencode.bdecode(file_modes.encode('utf-8')))