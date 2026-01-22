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
def import_unusual_file_modes(self, rev, unusual_file_modes):
    if unusual_file_modes:
        ret = [(path, unusual_file_modes[path]) for path in sorted(unusual_file_modes.keys())]
        rev.properties['file-modes'] = bencode.bencode(ret)