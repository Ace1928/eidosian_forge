from breezy import branch, errors
from breezy.tests.per_repository_reference import \
def create_stacked_branch(self):
    builder = self.make_branch_builder('source', format=self.bzrdir_format)
    builder.start_series()
    repo = builder.get_branch().repository
    if not repo._format.supports_external_lookups:
        raise tests.TestNotApplicable('format does not support stacking')
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'contents\n'))], revision_id=b'A-id')
    builder.build_snapshot([b'A-id'], [('modify', ('file', b'new-content\n'))], revision_id=b'B-id')
    builder.build_snapshot([b'B-id'], [('modify', ('file', b'yet more content\n'))], revision_id=b'C-id')
    builder.finish_series()
    source_b = builder.get_branch()
    source_b.lock_read()
    self.addCleanup(source_b.unlock)
    base = self.make_branch('base')
    base.pull(source_b, stop_revision=b'B-id')
    stacked = self.make_branch('stacked')
    stacked.set_stacked_on_url('../base')
    stacked.pull(source_b, stop_revision=b'C-id')
    return (base, stacked)