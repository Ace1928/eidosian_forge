from dulwich.tests import SkipTest, TestCase
from dulwich.tests.compat import utils
def test_git_version_none(self):
    self._version_str = b'not a git version'
    self.assertEqual(None, utils.git_version())