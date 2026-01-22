from breezy import errors, revision, tests
from breezy.tests import per_branch
def make_builder_with_merges(self, relpath):
    try:
        builder = self.make_branch_builder(relpath)
    except (errors.TransportNotPossible, errors.UninitializableFormat):
        raise tests.TestNotApplicable('format not directly constructable')
    builder.start_series()
    self.make_snapshot(builder, None, '1')
    self.make_snapshot(builder, ['1'], '1.1.1')
    self.make_snapshot(builder, ['1'], '2')
    self.make_snapshot(builder, ['2', '1.1.1'], '3')
    builder.finish_series()
    return builder