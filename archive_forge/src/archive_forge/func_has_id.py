import os
from .... import errors
from .... import transport as _mod_transport
from ....bzr import versionedfile
from ....errors import BzrError, UnlistableStore
from ....trace import mutter
def has_id(self, fileid, suffix=None):
    """See Store.has_id."""
    return self._transport.has_any(self._id_to_names(fileid, suffix))