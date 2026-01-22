from breezy import branch
from breezy.bzr import vf_search
from breezy.tests.per_repository import TestCaseWithRepository
def make_stacked_source_repo(self):
    _, source_b = self.make_source_branch()
    stack_b = self.make_branch('stack-on')
    stack_b.pull(source_b, stop_revision=b'B-id')
    stacked_b = self.make_branch('stacked')
    stacked_b.set_stacked_on_url('../stack-on')
    stacked_b.pull(source_b, stop_revision=b'C-id')
    return stacked_b.repository