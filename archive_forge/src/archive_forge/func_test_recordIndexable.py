import os
from operator import getitem
from twisted.python.compat import _PYPY
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.trial.unittest import TestCase
def test_recordIndexable(self):
    """
        The shadow user record returned by I{getpwnam} and I{getspall} is
        indexable, with successive indexes starting from 0 corresponding to the
        values of the C{sp_nam}, C{sp_pwd}, C{sp_lstchg}, C{sp_min}, C{sp_max},
        C{sp_warn}, C{sp_inact}, C{sp_expire}, and C{sp_flag} attributes,
        respectively.
        """
    db = self.database
    username, password, lastChange, min, max, warn, inact, expire, flag = self.getExistingUserInfo()
    for entry in [db.getspnam(username), db.getspall()[0]]:
        self.assertEqual(entry[0], username)
        self.assertEqual(entry[1], password)
        self.assertEqual(entry[2], lastChange)
        self.assertEqual(entry[3], min)
        self.assertEqual(entry[4], max)
        self.assertEqual(entry[5], warn)
        self.assertEqual(entry[6], inact)
        self.assertEqual(entry[7], expire)
        self.assertEqual(entry[8], flag)
        self.assertEqual(len(entry), len(list(entry)))
        self.assertRaises(IndexError, getitem, entry, 9)