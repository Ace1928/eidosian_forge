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
def test_use_site_url_if_endpoint_unset_v3(self):
    self.config_fixture.config(public_endpoint=None)
    for app in (self.public_app,):
        client = TestClient(app)
        resp = client.get('/v3/')
        self.assertEqual(http.client.OK, resp.status_int)
        data = jsonutils.loads(resp.body)
        expected = v3_VERSION_RESPONSE
        self._paste_in_port(expected['version'], 'http://localhost/v3/')
        self.assertEqual(expected, data)