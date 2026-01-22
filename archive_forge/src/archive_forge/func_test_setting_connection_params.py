import abc
import collections
import urllib
import uuid
from keystoneauth1 import _utils
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_setting_connection_params(self):
    text = uuid.uuid4().hex
    self.stub_url('GET', base_url=self.auth.url('prefix'), text=text)
    resp = self.session.get('prefix', endpoint_filter=self.ENDPOINT_FILTER)
    self.assertEqual(text, resp.text)
    self.assertEqual(self.auth.cert, self.requests_mock.last_request.cert)
    self.assertFalse(self.requests_mock.last_request.verify)