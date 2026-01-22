from tests.compat import mock, unittest
import datetime
import hashlib
import hmac
import locale
import time
import boto.utils
from boto.utils import Password
from boto.utils import pythonize_name
from boto.utils import _build_instance_metadata_url
from boto.utils import get_instance_userdata
from boto.utils import retry_url
from boto.utils import LazyLoadMetadata
from boto.compat import json, _thread
def clstest(self, cls):
    """Insure that password.__eq__ hashes test value before compare."""
    password = cls('foo')
    self.assertNotEquals(password, 'foo')
    password.set('foo')
    hashed = str(password)
    self.assertEquals(password, 'foo')
    self.assertEquals(password.str, hashed)
    password = cls(hashed)
    self.assertNotEquals(password.str, 'foo')
    self.assertEquals(password, 'foo')
    self.assertEquals(password.str, hashed)