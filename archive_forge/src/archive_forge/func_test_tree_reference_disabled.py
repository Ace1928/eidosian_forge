from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_tree_reference_disabled(self):
    old_inv = Inventory(None)
    new_inv = Inventory(None)
    root = new_inv.make_entry('directory', '', None, b'TREE_ROOT')
    root.revision = b'a@e\xc3\xa5ample.com--2004'
    new_inv.add(root)
    non_root = new_inv.make_entry('tree-reference', 'foo', root.file_id, b'id')
    non_root.revision = b'changed'
    non_root.reference_revision = b'subtree-version'
    new_inv.add(non_root)
    delta = new_inv._make_delta(old_inv)
    serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=True, tree_references=False)
    err = self.assertRaises(KeyError, serializer.delta_to_lines, NULL_REVISION, b'entry-version', delta)
    self.assertEqual(('tree-reference',), err.args)