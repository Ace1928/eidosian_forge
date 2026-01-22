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
def fix_person_identifier(text):
    if b'<' not in text and b'>' not in text:
        username = text
        email = text
    elif b'>' not in text:
        return text + b'>'
    else:
        if text.rindex(b'>') < text.rindex(b'<'):
            raise ValueError(text)
        username, email = text.split(b'<', 2)[-2:]
        email = email.split(b'>', 1)[0]
        if username.endswith(b' '):
            username = username[:-1]
    return b'%s <%s>' % (username, email)