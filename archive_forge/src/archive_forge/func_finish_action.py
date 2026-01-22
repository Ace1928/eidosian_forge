from .... import errors
from .... import transport as _mod_transport
from .... import ui
from ....diff import internal_diff
from ....revision import NULL_REVISION
from ....textfile import text_file
from ....timestamp import format_highres_date
from ....trace import mutter
from ...testament import StrictTestament
from ..bundle_data import BundleInfo, RevisionInfo
from . import BundleSerializer, _get_bundle_header, binary_diff
def finish_action(action, file_id, kind, meta_modified, text_modified, old_path, new_path):
    entry = new_tree.root_inventory.get_entry(file_id)
    if entry.revision != default_revision_id:
        action.add_utf8_property('last-changed', entry.revision)
    if meta_modified:
        action.add_bool_property('executable', entry.executable)
    if text_modified and kind == 'symlink':
        action.add_property('target', entry.symlink_target)
    if text_modified and kind == 'file':
        do_diff(file_id, old_path, new_path, action, force_binary)
    else:
        action.write(self.to_file)