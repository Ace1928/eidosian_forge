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
def test_setting_bad_connection_params(self):
    name = uuid.uuid4().hex
    self.auth.connection_params[name] = uuid.uuid4().hex
    e = self.assertRaises(exceptions.UnsupportedParameters, self.session.get, 'prefix', endpoint_filter=self.ENDPOINT_FILTER)
    self.assertIn(name, str(e))