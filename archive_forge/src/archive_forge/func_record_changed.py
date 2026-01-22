from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def record_changed(self, path, ie, parent_id=None):
    self._add_entry((path, path, ie.file_id, ie))
    self._modified_file_ids[path] = ie.file_id