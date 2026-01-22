import ipaddress
from openstack import exceptions
from openstack.tests.functional import base
def test_create_router_advanced(self):
    self._create_and_verify_advanced_router(external_cidr=u'10.2.2.0/24')