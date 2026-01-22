from breezy import tests, urlutils
from breezy.bzr import remote
from breezy.tests.per_repository import TestCaseWithRepository
def make_stacked_branch_with_long_history(self):
    builder = self.make_branch_builder('source')
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None))], revision_id=b'A')
    builder.build_snapshot([b'A'], [], revision_id=b'B')
    builder.build_snapshot([b'B'], [], revision_id=b'C')
    builder.build_snapshot([b'C'], [], revision_id=b'D')
    builder.build_snapshot([b'D'], [], revision_id=b'E')
    builder.build_snapshot([b'E'], [], revision_id=b'F')
    source_b = builder.get_branch()
    master_b = self.make_branch('master')
    master_b.pull(source_b, stop_revision=b'E')
    stacked_b = self.make_branch('stacked')
    stacked_b.set_stacked_on_url('../master')
    stacked_b.pull(source_b, stop_revision=b'F')
    builder.finish_series()
    return (master_b, stacked_b)