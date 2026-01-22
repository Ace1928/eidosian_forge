import re
import uuid
from keystoneauth1 import fixture
from oslo_serialization import jsonutils
from testtools import matchers
from keystoneclient import _discover
from keystoneclient.auth import token_endpoint
from keystoneclient import client
from keystoneclient import discover
from keystoneclient import exceptions
from keystoneclient import session
from keystoneclient.tests.unit import utils
from keystoneclient.v2_0 import client as v2_client
from keystoneclient.v3 import client as v3_client
def test_available_cinder_data(self):
    text = jsonutils.dumps(CINDER_EXAMPLES)
    self.requests_mock.get(BASE_URL, status_code=300, text=text)
    v1_url = '%sv1/' % BASE_URL
    v2_url = '%sv2/' % BASE_URL
    with self.deprecations.expect_deprecations_here():
        disc = discover.Discover(auth_url=BASE_URL)
    versions = disc.version_data()
    self.assertEqual((1, 0), versions[0]['version'])
    self.assertEqual('CURRENT', versions[0]['raw_status'])
    self.assertEqual(v1_url, versions[0]['url'])
    self.assertEqual((2, 0), versions[1]['version'])
    self.assertEqual('CURRENT', versions[1]['raw_status'])
    self.assertEqual(v2_url, versions[1]['url'])
    version = disc.data_for('v2.0')
    self.assertEqual((2, 0), version['version'])
    self.assertEqual('CURRENT', version['raw_status'])
    self.assertEqual(v2_url, version['url'])
    version = disc.data_for(1)
    self.assertEqual((1, 0), version['version'])
    self.assertEqual('CURRENT', version['raw_status'])
    self.assertEqual(v1_url, version['url'])
    self.assertIsNone(disc.url_for('v3'))
    self.assertEqual(v2_url, disc.url_for('v2'))
    self.assertEqual(v1_url, disc.url_for('v1'))