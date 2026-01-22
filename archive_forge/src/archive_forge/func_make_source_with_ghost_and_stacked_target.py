from breezy import branch
from breezy.bzr import vf_search
from breezy.tests.per_repository import TestCaseWithRepository
def make_source_with_ghost_and_stacked_target(self):
    builder = self.make_branch_builder('source')
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'content\n'))], revision_id=b'A-id')
    builder.build_snapshot([b'A-id', b'ghost-id'], [], revision_id=b'B-id')
    builder.finish_series()
    source_b = builder.get_branch()
    source_b.lock_read()
    self.addCleanup(source_b.unlock)
    base = self.make_branch('base')
    base.pull(source_b, stop_revision=b'A-id')
    stacked = self.make_branch('stacked')
    stacked.set_stacked_on_url('../base')
    return (source_b, base, stacked)