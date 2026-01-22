from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils
from ... import revision as _mod_revision
from ...bzr import inventory
from ...bzr.inventorytree import InventoryTreeChange
def path2id(self, path):
    try:
        return self._new_info_by_path[path].file_id
    except KeyError:
        return self._basis_inv.path2id(path)