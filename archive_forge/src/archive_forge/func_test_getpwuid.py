import os
from operator import getitem
from twisted.python.compat import _PYPY
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.trial.unittest import TestCase
def test_getpwuid(self):
    """
        I{getpwuid} accepts a uid and returns the user record associated with
        it.
        """
    for i in range(2):
        username, password, uid, gid, gecos, dir, shell = self.getExistingUserInfo()
        entry = self.database.getpwuid(uid)
        self.assertEqual(entry.pw_name, username)
        self.assertEqual(entry.pw_passwd, password)
        self.assertEqual(entry.pw_uid, uid)
        self.assertEqual(entry.pw_gid, gid)
        self.assertEqual(entry.pw_gecos, gecos)
        self.assertEqual(entry.pw_dir, dir)
        self.assertEqual(entry.pw_shell, shell)