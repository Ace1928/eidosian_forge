from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_uses_first_return(self):

    def get_tag_name_1(br, revid):
        return 'foo1'

    def get_tag_name_2(br, revid):
        return 'foo2'
    branch.Branch.hooks.install_named_hook('automatic_tag_name', get_tag_name_1, 'tagname1')
    branch.Branch.hooks.install_named_hook('automatic_tag_name', get_tag_name_2, 'tagname2')
    self.assertEqual('foo1', self.branch.automatic_tag_name(self.branch.last_revision()))