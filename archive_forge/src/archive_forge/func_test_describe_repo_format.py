import sys
from .. import branch as _mod_branch
from .. import controldir, errors, info
from .. import repository as _mod_repository
from .. import tests, workingtree
from ..bzr import branch as _mod_bzrbranch
def test_describe_repo_format(self):
    for key in controldir.format_registry.keys():
        if key in controldir.format_registry.aliases():
            continue
        if controldir.format_registry.get_info(key).hidden:
            continue
        expected = None
        if key in ('dirstate', 'knit', 'dirstate-tags'):
            expected = 'dirstate or dirstate-tags or knit'
        elif key in ('1.14',):
            expected = '1.14'
        elif key in ('1.14-rich-root',):
            expected = '1.14-rich-root'
        self.assertRepoDescription(key, expected)
    format = controldir.format_registry.make_controldir('knit')
    format.set_branch_format(_mod_bzrbranch.BzrBranchFormat6())
    tree = self.make_branch_and_tree('unknown', format=format)
    self.assertEqual('unnamed', info.describe_format(tree.controldir, tree.branch.repository, tree.branch, tree))