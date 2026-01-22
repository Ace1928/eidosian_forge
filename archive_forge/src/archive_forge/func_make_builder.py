from breezy import merge, tests
from breezy.plugins.changelog_merge import changelog_merge
from breezy.tests import test_merge_core
def make_builder(self):
    builder = test_merge_core.MergeBuilder(self.test_base_dir)
    self.addCleanup(builder.cleanup)
    return builder