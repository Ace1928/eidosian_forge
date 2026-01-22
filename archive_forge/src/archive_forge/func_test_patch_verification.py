import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def test_patch_verification(self):
    tree_a, tree_b, branch_c = self.make_trees()
    md = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 60, tree_b.branch.base, patch_type='bundle', public_branch=tree_a.branch.base)
    lines = md.to_lines()
    md2 = merge_directive.MergeDirective.from_lines(lines)
    md2._verify_patch(tree_a.branch.repository)
    md2.patch = md2.patch.replace(b' \n', b'\n')
    md2._verify_patch(tree_a.branch.repository)
    md2.patch = re.sub(b'(\r\n|\r|\n)', b'\r', md2.patch)
    self.assertTrue(md2._verify_patch(tree_a.branch.repository))
    md2.patch = re.sub(b'(\r\n|\r|\n)', b'\r\n', md2.patch)
    self.assertTrue(md2._verify_patch(tree_a.branch.repository))
    md2.patch = md2.patch.replace(b'content_c', b'content_d')
    self.assertFalse(md2._verify_patch(tree_a.branch.repository))