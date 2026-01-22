from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def record_rename(self, old_path, new_path, file_id, old_ie):
    new_ie = old_ie.copy()
    new_basename, new_parent_id = self._ensure_directory(new_path, self.basis_inventory)
    new_ie.name = new_basename
    new_ie.parent_id = new_parent_id
    new_ie.revision = self.revision_id
    self._add_entry((old_path, new_path, file_id, new_ie))
    self._modified_file_ids[new_path] = file_id
    self._paths_deleted_this_commit.discard(new_path)
    if new_ie.kind == 'directory':
        self.directory_entries[new_path] = new_ie