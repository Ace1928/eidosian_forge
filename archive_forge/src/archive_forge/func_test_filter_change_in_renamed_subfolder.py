from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_filter_change_in_renamed_subfolder(self):
    inv = Inventory(b'tree-root')
    inv.root.revision = b'rootrev'
    src_ie = inv.add_path('src', 'directory', b'src-id')
    src_ie.revision = b'srcrev'
    sub_ie = inv.add_path('src/sub/', 'directory', b'sub-id')
    sub_ie.revision = b'subrev'
    a_ie = inv.add_path('src/sub/a', 'file', b'a-id')
    a_ie.revision = b'filerev'
    a_ie.text_sha1 = osutils.sha_string(b'content\n')
    a_ie.text_size = len(b'content\n')
    chk_bytes = self.get_chk_bytes()
    inv = CHKInventory.from_inventory(chk_bytes, inv)
    inv = inv.create_by_apply_delta([('src/sub/a', 'src/sub/a', b'a-id', a_ie), ('src', 'src2', b'src-id', src_ie)], b'new-rev-2')
    new_inv = inv.filter([b'a-id', b'src-id'])
    self.assertEqual([('', b'tree-root'), ('src', b'src-id'), ('src/sub', b'sub-id'), ('src/sub/a', b'a-id')], [(path, ie.file_id) for path, ie in new_inv.iter_entries()])