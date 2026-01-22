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
def remove_readonly_transport_decorator(transport):
    if transport.is_readonly():
        try:
            return transport._decorated
        except AttributeError:
            raise bzr_errors.ReadOnlyError(transport)
    return transport