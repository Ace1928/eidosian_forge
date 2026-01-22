import os
from operator import getitem
from twisted.python.compat import _PYPY
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.trial.unittest import TestCase
def test_getspnamBytes(self):
    """
        I{getspnam} raises L{TypeError} when passed a L{bytes}, just like
        L{spwd.getspnam}.
        """
    self.assertRaises(TypeError, self.database.getspnam, b'i-am-bytes')