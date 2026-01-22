import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def test_generate_patch(self):
    tree_a, tree_b, branch_c = self.make_trees()
    md2 = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 120, tree_b.branch.base, patch_type='diff', public_branch=tree_a.branch.base)
    self.assertNotContainsRe(md2.patch, b'Bazaar revision bundle')
    self.assertContainsRe(md2.patch, b'\\+content_c')
    self.assertNotContainsRe(md2.patch, b'\\+\\+\\+ b/')
    self.assertContainsRe(md2.patch, b'\\+\\+\\+ file')