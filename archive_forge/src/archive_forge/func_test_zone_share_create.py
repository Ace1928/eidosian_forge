from openstack.dns.v2 import _proxy
from openstack.dns.v2 import floating_ip
from openstack.dns.v2 import recordset
from openstack.dns.v2 import zone
from openstack.dns.v2 import zone_export
from openstack.dns.v2 import zone_import
from openstack.dns.v2 import zone_share
from openstack.dns.v2 import zone_transfer
from openstack.tests.unit import test_proxy_base
def test_zone_share_create(self):
    self.verify_create(self.proxy.create_zone_share, zone_share.ZoneShare, method_kwargs={'zone': 'bogus_id'}, expected_kwargs={'zone_id': 'bogus_id'})