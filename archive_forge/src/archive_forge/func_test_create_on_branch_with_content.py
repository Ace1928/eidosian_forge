from ... import errors
from ...transport import NoSuchFile
from . import TestCaseWithTransport
def test_create_on_branch_with_content(self):
    """Creating a mutable tree on a non-trivial branch works."""
    wt = self.make_branch_and_tree('sometree')
    self.build_tree(['sometree/foo'])
    wt.add(['foo'])
    rev_id = wt.commit('first post')
    tree = wt.branch.create_memorytree()
    with tree.lock_read():
        self.assertEqual([rev_id], tree.get_parent_ids())
        self.assertEqual(b'contents of sometree/foo\n', tree.get_file('foo').read())