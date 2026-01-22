import os
from .... import errors, osutils
from .... import transport as _mod_transport
from .... import ui
from ....trace import mutter
from . import TransportStore
def get_empty(self, file_id, transaction):
    """Get an empty weave, which implies deleting the existing one first."""
    if self.has_id(file_id):
        self.delete(file_id, transaction)
    return self.get_weave_or_empty(file_id, transaction)