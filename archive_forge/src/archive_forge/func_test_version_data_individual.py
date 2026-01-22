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
def test_version_data_individual(self):
    mock = self.requests_mock.get(V3_URL, status_code=200, json=V3_VERSION_ENTRY)
    disc = discover.Discover(self.session, V3_URL)
    raw_data = disc.raw_version_data()
    clean_data = disc.version_data()
    for v in raw_data:
        self.assertEqual(v['id'], 'v3.0')
        self.assertEqual(v['status'], 'stable')
        self.assertIn('media-types', v)
        self.assertIn('links', v)
    for v in clean_data:
        self.assertEqual(v['version'], (3, 0))
        self.assertEqual(v['status'], discover.Status.CURRENT)
        self.assertEqual(v['raw_status'], 'stable')
        self.assertEqual(v['url'], V3_URL)
    self.assertTrue(mock.called_once)