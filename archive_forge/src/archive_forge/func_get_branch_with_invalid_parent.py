from breezy import branch, errors
from breezy.tests import per_branch, test_server
def get_branch_with_invalid_parent(self):
    """Get a branch whose get_parent will raise InaccessibleParent."""
    self.build_tree(['parent/', 'parent/path/', 'parent/path/to/', 'child/', 'child/path/', 'child/path/to/'], transport=self.get_transport())
    self.make_branch('parent/path/to/a').controldir.sprout(self.get_url('child/path/to/b'))
    self.get_transport().rename('child/path/to/b', 'b')
    branch_b = branch.Branch.open(self.get_readonly_url('b'))
    return branch_b