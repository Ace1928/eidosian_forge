from breezy import branch, config, errors, tests
from ..test_bedding import override_whoami
def test_whoami_remote_directory(self):
    """Test --directory option with a remote directory."""
    wt = self.make_branch_and_tree('subdir')
    self.set_branch_email(wt.branch, 'Branch Identity <branch@identi.ty>')
    url = self.get_readonly_url() + '/subdir'
    self.assertWhoAmI('Branch Identity <branch@identi.ty>', '--directory', url)
    url = self.get_url('subdir')
    self.run_bzr(['whoami', '--directory', url, '--branch', 'Changed Identity <changed@identi.ty>'])
    c = branch.Branch.open(url).get_config_stack()
    self.assertEqual('Changed Identity <changed@identi.ty>', c.get('email'))
    override_whoami(self)
    global_conf = config.GlobalStack()
    self.assertRaises(errors.NoWhoami, global_conf.get, 'email')