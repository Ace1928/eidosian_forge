import os
from operator import getitem
from twisted.python.compat import _PYPY
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.trial.unittest import TestCase
def test_noSuchUID(self):
    """
        I{getpwuid} raises L{KeyError} when passed a uid which does not exist
        in the user database.
        """
    self.assertRaises(KeyError, self.database.getpwuid, INVALID_UID)