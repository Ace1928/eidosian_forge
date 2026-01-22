import json
import re
from unittest import mock
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import http_basic
from keystoneauth1 import noauth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_version_data_override_version_url(self):
    self.requests_mock.get(V3_URL, status_code=200, json={'version': fixture.V3Discovery('http://override/identity/v3')})
    disc = discover.Discover(self.session, V3_URL)
    version_data = disc.version_data()
    for v in version_data:
        self.assertEqual(v['version'], (3, 0))
        self.assertEqual(v['status'], discover.Status.CURRENT)
        self.assertEqual(v['raw_status'], 'stable')
        self.assertEqual(v['url'], V3_URL)
    self.requests_mock.get(BASE_URL, status_code=200, json={'version': fixture.V3Discovery('http://override/identity/v3')})
    disc = discover.Discover(self.session, BASE_URL)
    version_data = disc.version_data()
    for v in version_data:
        self.assertEqual(v['version'], (3, 0))
        self.assertEqual(v['status'], discover.Status.CURRENT)
        self.assertEqual(v['raw_status'], 'stable')
        self.assertEqual(v['url'], V3_URL)