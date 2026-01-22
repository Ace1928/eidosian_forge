from breezy import branch, config, errors, tests
from ..test_bedding import override_whoami
def test_whoami_directory(self):
    """Test --directory option."""
    wt = self.make_branch_and_tree('subdir')
    self.set_branch_email(wt.branch, 'Branch Identity <branch@identi.ty>')
    self.assertWhoAmI('Branch Identity <branch@identi.ty>', '--directory', 'subdir')
    self.run_bzr(['whoami', '--directory', 'subdir', '--branch', 'Changed Identity <changed@identi.ty>'])
    wt = wt.controldir.open_workingtree()
    c = wt.branch.get_config_stack()
    self.assertEqual('Changed Identity <changed@identi.ty>', c.get('email'))