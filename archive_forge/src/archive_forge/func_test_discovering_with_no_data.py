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
def test_discovering_with_no_data(self):
    self.stub_url('GET', [], base_url=self.TEST_COMPUTE_ADMIN, status_code=400)
    body = 'SUCCESS'
    self.stub_url('GET', ['path'], base_url=self.TEST_COMPUTE_ADMIN, text=body, status_code=200)
    a = self.create_auth_plugin()
    s = session.Session(auth=a)
    resp = s.get('/path', endpoint_filter={'service_type': 'compute', 'interface': 'admin', 'version': self.version})
    self.assertEqual(200, resp.status_code)
    self.assertEqual(body, resp.text)