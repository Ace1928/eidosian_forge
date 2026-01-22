from breezy import errors, revision, tests
from breezy.tests import per_branch
def make_branch_with_alternate_ancestries(self, relpath='.'):
    builder = self.make_branch_builder(relpath)
    builder.start_series()
    self.make_snapshot(builder, None, '1')
    self.make_snapshot(builder, ['1'], '1.1.1')
    self.make_snapshot(builder, ['1', '1.1.1'], '2')
    self.make_snapshot(builder, ['1.1.1'], '1.2.1')
    self.make_snapshot(builder, ['1.1.1', '1.2.1'], '1.1.2')
    self.make_snapshot(builder, ['2', '1.1.2'], '3')
    builder.finish_series()
    br = builder.get_branch()
    br.lock_read()
    self.addCleanup(br.unlock)
    return br