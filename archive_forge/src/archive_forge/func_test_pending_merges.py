import os
from ....branch import Branch
from ....tests.blackbox import ExternalBase
def test_pending_merges(self):
    os.chdir('..')
    self.build_tree_contents([('main/hello', '42')])
    self.run_bzr('add', working_dir='main')
    self.run_bzr('commit -m that main')
    self.build_tree_contents([('feature/hoi', 'my data')])
    self.run_bzr('add', working_dir='feature')
    self.run_bzr('commit -m this feature')
    self.assertEqual(('', ' M  hello\nAll changes applied successfully.\n'), self.run_bzr('merge ../main', working_dir='feature'))
    out, err = self.run_bzr('rebase --pending-merges', working_dir='feature')
    self.assertEqual('', out)
    self.assertContainsRe(err, 'modified hello')
    self.assertEqual(('3\n', ''), self.run_bzr('revno', working_dir='feature'))