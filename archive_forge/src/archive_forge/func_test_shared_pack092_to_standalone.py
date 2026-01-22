from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_shared_pack092_to_standalone(self):
    self.test_shared_format_to_standalone('pack-0.92')