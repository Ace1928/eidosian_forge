import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def test_broken_bundle(self):
    tree_a, tree_b, branch_c = self.make_trees()
    md1 = self.from_objects(tree_a.branch.repository, b'rev2a', 500, 120, tree_b.branch.base, public_branch=branch_c.base)
    lines = md1.to_lines()
    lines = [l.replace(b'\n', b'\r\n') for l in lines]
    md2 = merge_directive.MergeDirective.from_lines(lines)
    self.assertEqual(b'rev2a', md2.revision_id)