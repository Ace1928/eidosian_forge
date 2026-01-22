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
def test_available_versions_basics(self):
    examples = {'keystone': V3_VERSION_LIST, 'cinder': jsonutils.dumps(CINDER_EXAMPLES), 'glance': jsonutils.dumps(GLANCE_EXAMPLES)}
    for path, text in examples.items():
        url = '%s%s' % (BASE_URL, path)
        self.requests_mock.get(url, status_code=300, text=text)
        versions = discover.available_versions(url)
        for v in versions:
            for n in ('id', 'status', 'links'):
                msg = '%s missing from %s version data' % (n, path)
                self.assertThat(v, matchers.Annotate(msg, matchers.Contains(n)))