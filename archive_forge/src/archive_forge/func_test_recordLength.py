import os
from operator import getitem
from twisted.python.compat import _PYPY
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.trial.unittest import TestCase
def test_recordLength(self):
    """
        The shadow user record returned by I{getspnam} and I{getspall} has a
        length.
        """
    db = self.database
    username = self.getExistingUserInfo()[0]
    for entry in [db.getspnam(username), db.getspall()[0]]:
        self.assertIsInstance(len(entry), int)
        self.assertEqual(len(entry), 9)