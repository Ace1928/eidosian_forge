from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def get_zone_share(self, zone, zone_share):
    """Get a zone share

        :param zone: The value can be the ID of a zone
            or a :class:`~openstack.dns.v2.zone.Zone` instance.
        :param zone_share: The zone_share can be either the ID of the zone
            share or a :class:`~openstack.dns.v2.zone_share.ZoneShare` instance
            that the zone share belongs to.

        :returns: ZoneShare instance.
        :rtype: :class:`~openstack.dns.v2.zone_share.ZoneShare`
        """
    zone_obj = self._get_resource(_zone.Zone, zone)
    return self._get(_zone_share.ZoneShare, zone_share, zone_id=zone_obj.id)