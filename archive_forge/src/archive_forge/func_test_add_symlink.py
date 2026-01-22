from .. import errors, transport
from ..memorytree import MemoryTree
from ..treebuilder import TreeBuilder
from . import TestCaseWithTransport
def test_add_symlink(self):
    branch = self.make_branch('branch')
    tree = MemoryTree.create_on_branch(branch)
    with tree.lock_write():
        tree._file_transport.symlink('bar', 'foo')
        tree.add(['', 'foo'])
        self.assertEqual('symlink', tree.kind('foo'))
        self.assertEqual('bar', tree.get_symlink_target('foo'))