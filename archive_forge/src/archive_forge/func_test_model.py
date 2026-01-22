import unittest
import logging
import time
def test_model(self, hashfunc=None):
    from boto.utils import Password
    from boto.sdb.db.model import Model
    from boto.sdb.db.property import PasswordProperty
    import hashlib

    class MyModel(Model):
        password = PasswordProperty(hashfunc=hashfunc)
    return MyModel