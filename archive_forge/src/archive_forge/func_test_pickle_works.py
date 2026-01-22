import copy
import pickle
import os
from tests.compat import unittest, mock
from tests.unit import MockServiceWithConfigTestCase
from nose.tools import assert_equal
from boto.auth import HmacAuthV4Handler
from boto.auth import S3HmacAuthV4Handler
from boto.auth import detect_potential_s3sigv4
from boto.auth import detect_potential_sigv4
from boto.connection import HTTPRequest
from boto.provider import Provider
from boto.regioninfo import RegionInfo
def test_pickle_works(self):
    provider = Provider('aws', access_key='access_key', secret_key='secret_key')
    auth = HmacAuthV4Handler('queue.amazonaws.com', None, provider)
    pickled = pickle.dumps(auth)
    auth2 = pickle.loads(pickled)
    self.assertEqual(auth.host, auth2.host)