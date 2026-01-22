from breezy.branch import UnstackableBranchFormat
from breezy.errors import IncompatibleFormat
from breezy.revision import NULL_REVISION
from breezy.tests import TestCaseWithTransport
def test__get_nick(self):
    """Make sure _get_nick is implemented and returns a string."""
    branch = self.make_branch()
    self.assertIsInstance(branch._get_nick(local=False), str)
    self.assertIsInstance(branch._get_nick(local=True), str)