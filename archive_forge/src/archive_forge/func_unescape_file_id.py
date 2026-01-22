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
def unescape_file_id(file_id):
    ret = bytearray()
    i = 0
    while i < len(file_id):
        if file_id[i:i + 1] != b'_':
            ret.append(file_id[i])
        else:
            if file_id[i + 1:i + 2] == b'_':
                ret.append(b'_'[0])
            elif file_id[i + 1:i + 2] == b's':
                ret.append(b' '[0])
            elif file_id[i + 1:i + 2] == b'c':
                ret.append(b'\x0c'[0])
            else:
                raise ValueError('unknown escape character %s' % file_id[i + 1:i + 2])
            i += 1
        i += 1
    return bytes(ret)