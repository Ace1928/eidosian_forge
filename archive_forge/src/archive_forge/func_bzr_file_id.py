from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def bzr_file_id(self, path):
    """Get a Bazaar file identifier for a path."""
    return self.bzr_file_id_and_new(path)[0]