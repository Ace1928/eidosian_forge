import sys
import breezy.errors
from breezy import urlutils
from breezy.osutils import getcwd
from breezy.tests import TestCaseWithTransport, TestNotApplicable, TestSkipped
def test_win32_set_parent_on_another_drive(self):
    if sys.platform != 'win32':
        raise TestSkipped('windows-specific test')
    b = self.make_branch('.')
    base_url = b.controldir.transport.abspath('.')
    if not base_url.startswith('file:///'):
        raise TestNotApplicable('this test should be run with local base')
    base = urlutils.local_path_from_url(base_url)
    other = 'file:///D:/path'
    if base[0] != 'C':
        other = 'file:///C:/path'
    b.set_parent(other)
    self.assertEqual(other, b._get_parent_location())