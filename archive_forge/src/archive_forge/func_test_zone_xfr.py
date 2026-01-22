from openstack.dns.v2 import _proxy
from openstack.dns.v2 import floating_ip
from openstack.dns.v2 import recordset
from openstack.dns.v2 import zone
from openstack.dns.v2 import zone_export
from openstack.dns.v2 import zone_import
from openstack.dns.v2 import zone_share
from openstack.dns.v2 import zone_transfer
from openstack.tests.unit import test_proxy_base
def test_zone_xfr(self):
    self._verify('openstack.dns.v2.zone.Zone.xfr', self.proxy.xfr_zone, method_args=[{'zone': 'id'}], expected_args=[self.proxy])