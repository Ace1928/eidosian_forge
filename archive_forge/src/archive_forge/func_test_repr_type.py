from breezy.branch import UnstackableBranchFormat
from breezy.errors import IncompatibleFormat
from breezy.revision import NULL_REVISION
from breezy.tests import TestCaseWithTransport
def test_repr_type(self):
    branch = self.make_branch()
    self.assertIsInstance(repr(branch), str)