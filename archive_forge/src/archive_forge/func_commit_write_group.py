import os
import threading
from dulwich.objects import ShaFile, hex_to_sha, sha_to_hex
from .. import bedding
from .. import errors as bzr_errors
from .. import osutils, registry, trace
from ..bzr import btree_index as _mod_btree_index
from ..bzr import index as _mod_index
from ..bzr import versionedfile
from ..transport import FileExists, NoSuchFile, get_transport_from_path
def commit_write_group(self):
    if self._builder is None:
        raise bzr_errors.BzrError('builder not open')
    stream = self._builder.finish()
    name = self._name.hexdigest() + '.rix'
    size = self._transport.put_file(name, stream)
    index = _mod_btree_index.BTreeGraphIndex(self._transport, name, size)
    self._index.insert_index(0, index)
    self._builder = None
    self._name = None