from openstack.dns.v2 import _proxy
from openstack.dns.v2 import floating_ip
from openstack.dns.v2 import recordset
from openstack.dns.v2 import zone
from openstack.dns.v2 import zone_export
from openstack.dns.v2 import zone_import
from openstack.dns.v2 import zone_share
from openstack.dns.v2 import zone_transfer
from openstack.tests.unit import test_proxy_base
def test_zone_share_get(self):
    self.verify_get(self.proxy.get_zone_share, zone_share.ZoneShare, method_args=['zone', 'zone_share'], expected_args=['zone_share'], expected_kwargs={'zone_id': 'zone'})