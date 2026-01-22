from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def do_checkout_test(self, lightweight):
    tree = self.make_branch_and_tree('source', format=self.get_format_name_subtree())
    subtree = self.make_branch_and_tree('source/subtree', format=self.get_format_name_subtree())
    subsubtree = self.make_branch_and_tree('source/subtree/subsubtree', format=self.get_format_name_subtree())
    self.build_tree(['source/subtree/file', 'source/subtree/subsubtree/file'])
    subsubtree.add('file')
    subtree.add('file')
    subtree.add_reference(subsubtree)
    subtree.set_reference_info('subsubtree', subsubtree.branch.user_url)
    tree.add_reference(subtree)
    tree.set_reference_info('subtree', subtree.branch.user_url)
    tree.commit('a revision')
    subtree.commit('a subtree file')
    subsubtree.commit('a subsubtree file')
    tree.branch.create_checkout('target', lightweight=lightweight)
    self.assertPathExists('target')
    self.assertPathExists('target/subtree')
    self.assertPathExists('target/subtree/file')
    self.assertPathExists('target/subtree/subsubtree/file')
    subbranch = _mod_branch.Branch.open('target/subtree/subsubtree')
    if lightweight:
        self.assertEndsWith(subbranch.base, 'source/subtree/subsubtree/')
    else:
        self.assertEndsWith(subbranch.base, 'target/subtree/subsubtree/')