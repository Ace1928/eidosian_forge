from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
def test_host_set_none(self):
    ret = self.api.host_set(host='myhost')
    self.assertIsNone(ret)