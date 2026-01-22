from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_shared_rich_root_pack_to_standalone(self):
    self.test_shared_format_to_standalone('rich-root-pack')