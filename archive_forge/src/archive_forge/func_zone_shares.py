from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def zone_shares(self, zone, **query):
    """Retrieve a generator of zone sharess

        :param zone: The zone ID or a
            :class:`~openstack.dns.v2.zone.Zone` instance
        :param dict query: Optional query parameters to be sent to limit the
                           resources being returned.

                           * `target_project_id`: The target project ID field.

        :returns: A generator of zone shares
            :class:`~openstack.dns.v2.zone_share.ZoneShare` instances.
        """
    zone_obj = self._get_resource(_zone.Zone, zone)
    return self._list(_zone_share.ZoneShare, zone_id=zone_obj.id, **query)