import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def get_tree_with_cachable_file_foo(self):
    tree = self.make_branch_and_tree('.')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    self.build_tree_contents([('foo', b'a bit of content for foo\n')])
    tree.add(['foo'], ids=[b'foo-id'])
    tree.current_dirstate()._cutoff_time = time.time() + 60
    return tree