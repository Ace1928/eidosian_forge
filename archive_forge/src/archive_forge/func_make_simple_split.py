from breezy import errors
from breezy.bzr import knit
from breezy.tests.per_repository_reference import \
def make_simple_split(self):
    """Set up the repositories so that everything is in base except F"""
    self.base_repo.fetch(self.all_repo, revision_id=b'G')
    self.stacked_repo.fetch(self.all_repo, revision_id=b'F')