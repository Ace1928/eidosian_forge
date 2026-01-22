from dulwich.tests import SkipTest, TestCase
from dulwich.tests.compat import utils
def run_git(args, **unused_kwargs):
    self.assertEqual(['--version'], args)
    return (0, self._version_str, '')