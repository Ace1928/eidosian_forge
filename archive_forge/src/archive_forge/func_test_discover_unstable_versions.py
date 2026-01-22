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
def test_discover_unstable_versions(self):
    version_list = fixture.DiscoveryList(BASE_URL, v3_status='beta')
    self.requests_mock.get(BASE_URL, status_code=300, json=version_list)
    self.assertCreatesV2(auth_url=BASE_URL)
    self.assertVersionNotAvailable(auth_url=BASE_URL, version=3)
    self.assertCreatesV3(auth_url=BASE_URL, unstable=True)