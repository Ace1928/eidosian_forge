from .. import branch, errors
from . import TestCaseWithTransport
def test_extract_in_deep_checkout(self):
    a_branch = self.make_branch('branch', format='rich-root-pack')
    self.build_tree(['a/', 'a/b/', 'a/b/c/', 'a/b/c/d/', 'a/b/c/d/e'])
    wt = a_branch.create_checkout('a', lightweight=True)
    wt.add(['b', 'b/c', 'b/c/d', 'b/c/d/e/'], ids=[b'b-id', b'c-id', b'd-id', b'e-id'])
    wt.commit('added files')
    b_wt = wt.extract('b/c/d')
    b_branch = branch.Branch.open('branch/b/c/d')
    b_branch_ref = branch.Branch.open('a/b/c/d')
    self.assertEqual(b_branch.base, b_branch_ref.base)