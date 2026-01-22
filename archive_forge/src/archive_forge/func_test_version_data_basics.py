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
def test_version_data_basics(self):
    examples = {'keystone': V3_VERSION_LIST, 'cinder': CINDER_EXAMPLES, 'glance': GLANCE_EXAMPLES}
    for path, data in examples.items():
        url = '%s%s' % (BASE_URL, path)
        mock = self.requests_mock.get(url, status_code=300, json=data)
        disc = discover.Discover(self.session, url)
        raw_data = disc.raw_version_data()
        clean_data = disc.version_data()
        for v in raw_data:
            for n in ('id', 'status', 'links'):
                msg = '%s missing from %s version data' % (n, path)
                self.assertThat(v, matchers.Annotate(msg, matchers.Contains(n)))
        for v in clean_data:
            for n in ('version', 'url', 'raw_status'):
                msg = '%s missing from %s version data' % (n, path)
                self.assertThat(v, matchers.Annotate(msg, matchers.Contains(n)))
        self.assertTrue(mock.called_once)