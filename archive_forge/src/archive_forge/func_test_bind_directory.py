from breezy import branch, controldir, errors, tests
from breezy.tests import script
def test_bind_directory(self):
    """Test --directory option"""
    tree = self.make_branch_and_tree('base')
    self.build_tree(['base/a', 'base/b'])
    tree.add('a', ids=b'b')
    tree.commit(message='init')
    branch = tree.branch
    tree.controldir.sprout('child')
    self.run_bzr('bind --directory=child base')
    d = controldir.ControlDir.open('child')
    self.assertNotEqual(None, d.open_branch().get_master_branch())
    self.run_bzr('unbind -d child')
    self.assertEqual(None, d.open_branch().get_master_branch())
    self.run_bzr('unbind --directory child', retcode=3)