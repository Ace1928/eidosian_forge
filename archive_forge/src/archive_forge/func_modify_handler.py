from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def modify_handler(self, filecmd):
    kind, executable = mode_to_kind(filecmd.mode)
    if filecmd.dataref is not None:
        if kind == 'directory':
            data = None
        elif kind == 'tree-reference':
            data = filecmd.dataref
        else:
            data = self.cache_mgr.fetch_blob(filecmd.dataref)
    else:
        data = filecmd.data
    self.debug('modifying %s', filecmd.path)
    decoded_path = self._decode_path(filecmd.path)
    self._modify_item(decoded_path, kind, executable, data, self.basis_inventory)