import datetime
import io
import itertools
import json
import logging
import sys
from unittest import mock
import uuid
from oslo_utils import encodeutils
import requests
import requests.auth
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneauth1 import session as client_session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_endpoint_override_fails_to_do_unknown_replacement(self):
    auth = CalledAuthPlugin()
    sess = client_session.Session(auth=auth)
    override_base = 'http://mytest/%(unknown_id)s'
    e = self.assertRaises(AttributeError, sess.get, '/path', endpoint_override=override_base, endpoint_filter={'service_type': 'identity'})
    self.assertIn('unknown_id', str(e))