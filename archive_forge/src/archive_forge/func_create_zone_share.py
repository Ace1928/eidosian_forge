from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def create_zone_share(self, zone, **attrs):
    """Create a new zone share from attributes

        :param zone: The zone ID or a
            :class:`~openstack.dns.v2.zone.Zone` instance
        :param dict attrs: Keyword arguments which will be used to create
            a :class:`~openstack.dns.v2.zone_share.ZoneShare`,
            comprised of the properties on the ZoneShare class.

        :returns: The results of zone share creation
        :rtype: :class:`~openstack.dns.v2.zone_share.ZoneShare`
        """
    zone_obj = self._get_resource(_zone.Zone, zone)
    return self._create(_zone_share.ZoneShare, zone_id=zone_obj.id, **attrs)