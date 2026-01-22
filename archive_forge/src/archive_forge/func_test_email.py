import re
from .. import errors, gpg, mail_client, merge_directive, tests, trace
def test_email(self):
    tree_a, tree_b, branch_c = self.make_trees()
    md = self.from_objects(tree_a.branch.repository, b'rev2a', 476, 60, tree_b.branch.base, patch_type=None, public_branch=tree_a.branch.base)
    message = md.to_email('pqm@example.com', tree_a.branch)
    self.assertContainsRe(message.as_string(), self.EMAIL1)
    md.message = 'Commit of rev2a with special message'
    message = md.to_email('pqm@example.com', tree_a.branch)
    self.assertContainsRe(message.as_string(), self.EMAIL2)