from breezy import errors
from breezy.tests.per_controldir import TestCaseWithControlDir
def test_supports_transport(self):
    self.assertIsInstance(self.bzrdir_format.supports_transport(self.get_transport()), bool)