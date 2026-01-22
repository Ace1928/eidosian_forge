import ddt
from keystoneauth1 import exceptions
from openstack.tests.unit import base
@ddt.data({}, {'raise_exc': False})
def test_discovery_err(self, get_kwargs):
    self._register_uris(status_code=500)
    rs = self.cloud.placement.get('/allocation_candidates', **get_kwargs)
    self._validate_resp(rs, 500)