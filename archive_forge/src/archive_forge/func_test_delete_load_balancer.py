from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_delete_load_balancer(self):
    self.requests_mock.register_uri('DELETE', FAKE_LBAAS_URL + 'loadbalancers/' + FAKE_LB, status_code=200)
    ret = self.api.load_balancer_delete(FAKE_LB)
    self.assertEqual(200, ret.status_code)