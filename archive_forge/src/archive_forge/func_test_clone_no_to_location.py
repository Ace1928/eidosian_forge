import os
from breezy import branch, controldir, tests
from breezy.urlutils import local_path_to_url
def test_clone_no_to_location(self):
    """The to_location is derived from the source branch name."""
    os.mkdir('something')
    a = self.example_dir('something/a').branch
    self.run_bzr('clone something/a')
    b = branch.Branch.open('a')
    self.assertEqual(b.last_revision_info(), a.last_revision_info())