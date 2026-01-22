import copy
import functools
import random
import http.client
from oslo_serialization import jsonutils
from testtools import matchers as tt_matchers
import webob
from keystone.api import discovery
from keystone.common import json_home
from keystone.tests import unit
def test_use_site_url_if_endpoint_unset(self):
    self.config_fixture.config(public_endpoint=None)
    for app in (self.public_app,):
        client = TestClient(app)
        resp = client.get('/')
        self.assertEqual(300, resp.status_int)
        data = jsonutils.loads(resp.body)
        expected = VERSIONS_RESPONSE
        for version in expected['versions']['values']:
            if version['id'].startswith('v3'):
                self._paste_in_port(version, 'http://localhost/v3/')
        self.assertThat(data, _VersionsEqual(expected))