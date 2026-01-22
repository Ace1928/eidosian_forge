import os
from breezy import branch, controldir, tests
from breezy.urlutils import local_path_to_url
def test_from_colocated(self):
    """Branch from a colocated branch into a regular branch."""
    os.mkdir('b')
    tree = self.example_dir('b/a')
    tree.controldir.create_branch(name='somecolo')
    out, err = self.run_bzr('clone %s' % local_path_to_url('b/a'))
    self.assertEqual('', out)
    self.assertEqual('Created new control directory.\n', err)
    self.assertPathExists('a')
    target = controldir.ControlDir.open('a')
    self.assertEqual(['', 'somecolo'], target.branch_names())