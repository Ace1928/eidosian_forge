from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session
from osc_lib.api import api
from osc_lib import exceptions
from osc_lib.tests.api import fakes as api_fakes
def test_baseapi_delete(self):
    self.requests_mock.register_uri('DELETE', self.BASE_URL + '/qaz', status_code=204)
    ret = self.api.delete('qaz')
    self.assertEqual(204, ret.status_code)