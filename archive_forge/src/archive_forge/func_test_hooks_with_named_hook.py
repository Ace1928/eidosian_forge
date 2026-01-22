from breezy.branch import Branch
from breezy.tests import TestCaseWithTransport
def test_hooks_with_named_hook(self):
    self.make_branch('.')

    def foo():
        return
    name = 'Foo Bar Hook'
    Branch.hooks.install_named_hook('post_push', foo, name)
    out, err = self.run_bzr('hooks')
    self._check_hooks_output(out, {'post_push': [name]})