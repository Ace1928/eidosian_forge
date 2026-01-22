import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
@staticmethod
def metadata_record(serializer, revision_id, message=None):
    metadata = {b'revision_id': revision_id}
    if message is not None:
        metadata[b'message'] = message.encode('utf-8')
    return serializer.bytes_record(bencode.bencode(metadata), ((b'metadata',),))