import os
from breezy.controldir import ControlDir
from breezy.filters import ContentFilter
from breezy.switch import switch
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def test_symmetric_content_filtering(self):
    tree, txt_path, bin_path = self.create_cf_tree(txt_reader=_swapcase, txt_writer=_swapcase)
    basis = tree.basis_tree()
    basis.lock_read()
    self.addCleanup(basis.unlock)
    if tree.supports_content_filtering():
        expected = b'fOO tXT'
    else:
        expected = b'Foo Txt'
    self.assertEqual(expected, basis.get_file_text(txt_path))
    self.assertEqual(b'Foo Bin', basis.get_file_text(bin_path))
    tree.lock_read()
    self.addCleanup(tree.unlock)
    with tree.get_file(txt_path, filtered=False) as f:
        self.assertEqual(b'Foo Txt', f.read())
    with tree.get_file(bin_path, filtered=False) as f:
        self.assertEqual(b'Foo Bin', f.read())