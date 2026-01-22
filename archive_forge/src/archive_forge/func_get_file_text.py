from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils
from ... import revision as _mod_revision
from ...bzr import inventory
from ...bzr.inventorytree import InventoryTreeChange
def get_file_text(self, path):
    file_id = self.path2id(path)
    try:
        return self._content_provider(file_id)
    except KeyError:
        assert file_id not in self._new_info_by_id
        old_ie = self._basis_inv.get_entry(file_id)
        old_text_key = (file_id, old_ie.revision)
        stream = self._repo.texts.get_record_stream([old_text_key], 'unordered', True)
        return next(stream).get_bytes_as('fulltext')