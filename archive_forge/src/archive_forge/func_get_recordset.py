from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def get_recordset(self, recordset, zone):
    """Get a recordset

        :param zone: The value can be the ID of a zone
            or a :class:`~openstack.dns.v2.zone.Zone` instance.
        :param recordset: The value can be the ID of a recordset
            or a :class:`~openstack.dns.v2.recordset.Recordset` instance.
        :returns: Recordset instance
        :rtype: :class:`~openstack.dns.v2.recordset.Recordset`
        """
    zone = self._get_resource(_zone.Zone, zone)
    return self._get(_rs.Recordset, recordset, zone_id=zone.id)