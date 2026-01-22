from dulwich.tests import SkipTest, TestCase
from dulwich.tests.compat import utils
def test_git_version_extra(self):
    self._version_str = b'git version 1.7.0.3.295.gd8fa2'
    self.assertEqual((1, 7, 0, 3), utils.git_version())