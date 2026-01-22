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
def test_hmac(self):

    def hmac_hashfunc(cls, msg):
        if not isinstance(msg, bytes):
            msg = msg.encode('utf-8')
        return hmac.new(b'mysecretkey', msg)

    class HMACPassword(Password):
        hashfunc = hmac_hashfunc
    self.clstest(HMACPassword)
    password = HMACPassword()
    password.set('foo')
    self.assertEquals(str(password), hmac.new(b'mysecretkey', b'foo').hexdigest())