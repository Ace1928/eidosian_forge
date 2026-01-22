import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def test_use_public_submit_branch(self):
    tree_a, tree_b, branch_c = self.make_trees()
    branch_c.pull(tree_a.branch)
    md = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 144, tree_b.branch.base, patch_type=None, public_branch=branch_c.base)
    self.assertEqual(md.target_branch, tree_b.branch.base)
    tree_b.branch.set_public_branch('http://example.com')
    md2 = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 144, tree_b.branch.base, patch_type=None, public_branch=branch_c.base)
    self.assertEqual(md2.target_branch, 'http://example.com')