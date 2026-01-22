from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def make_simple_inventory(self):
    inv = Inventory(b'TREE_ROOT')
    inv.revision_id = b'revid'
    inv.root.revision = b'rootrev'
    self.make_dir(inv, 'dir1', b'TREE_ROOT', b'dirrev')
    self.make_dir(inv, 'dir2', b'TREE_ROOT', b'dirrev')
    self.make_dir(inv, 'sub-dir1', b'dir1-id', b'dirrev')
    self.make_file(inv, 'top', b'TREE_ROOT', b'filerev')
    self.make_file(inv, 'sub-file1', b'dir1-id', b'filerev')
    self.make_file(inv, 'sub-file2', b'dir1-id', b'filerev')
    self.make_file(inv, 'subsub-file1', b'sub-dir1-id', b'filerev')
    self.make_file(inv, 'sub2-file1', b'dir2-id', b'filerev')
    chk_bytes = self.get_chk_bytes()
    chk_inv = CHKInventory.from_inventory(chk_bytes, inv, maximum_size=100, search_key_name=b'hash-255-way')
    lines = chk_inv.to_lines()
    return CHKInventory.deserialise(chk_bytes, lines, (b'revid',))