from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def get_floating_ip(self, floating_ip):
    """Get a Floating IP

        :param floating_ip: The value can be the ID of a floating ip
            or a :class:`~openstack.dns.v2.floating_ip.FloatingIP` instance.
            The ID is in format "region_name:floatingip_id"
        :returns: FloatingIP instance.
        :rtype: :class:`~openstack.dns.v2.floating_ip.FloatingIP`
        """
    return self._get(_fip.FloatingIP, floating_ip)