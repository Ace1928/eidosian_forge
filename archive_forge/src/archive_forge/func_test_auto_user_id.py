import os
import sys
from .. import bedding, osutils, tests
def test_auto_user_id(self):
    """Automatic inference of user name.

        This is a bit hard to test in an isolated way, because it depends on
        system functions that go direct to /etc or perhaps somewhere else.
        But it's reasonable to say that on Unix, with an /etc/mailname, we ought
        to be able to choose a user name with no configuration.
        """
    if sys.platform == 'win32':
        raise tests.TestSkipped('User name inference not implemented on win32')
    realname, address = bedding._auto_user_id()
    if os.path.exists('/etc/mailname'):
        self.assertIsNot(None, realname)
        self.assertIsNot(None, address)
    else:
        self.assertEqual((None, None), (realname, address))