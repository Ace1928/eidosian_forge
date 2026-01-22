import os
from breezy import ignores
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import run_script
def test_clean_tree_interactive(self):
    wt = self.make_branch_and_tree('.')
    self.touch('bar')
    self.touch('foo')
    run_script(self, '\n        $ brz clean-tree\n        bar\n        foo\n        2>Are you sure you wish to delete these? ([y]es, [n]o): no\n        <n\n        Canceled\n        ')
    self.assertPathExists('bar')
    self.assertPathExists('foo')
    run_script(self, '\n        $ brz clean-tree\n        bar\n        foo\n        2>Are you sure you wish to delete these? ([y]es, [n]o): yes\n        <y\n        2>deleting paths:\n        2>  bar\n        2>  foo\n        ')
    self.assertPathDoesNotExist('bar')
    self.assertPathDoesNotExist('foo')