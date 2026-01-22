from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def create_zone_export(self, zone, **attrs):
    """Create a new zone export from attributes

        :param zone: The value can be the ID of a zone to be exported
            or a :class:`~openstack.dns.v2.zone_export.ZoneExport` instance.
        :param dict attrs: Keyword arguments which will be used to create
            a :class:`~openstack.dns.v2.zone_export.ZoneExport`,
            comprised of the properties on the ZoneExport class.
        :returns: The results of zone creation.
        :rtype: :class:`~openstack.dns.v2.zone_export.ZoneExport`
        """
    zone = self._get_resource(_zone.Zone, zone)
    return self._create(_zone_export.ZoneExport, base_path='/zones/%(zone_id)s/tasks/export', prepend_key=False, zone_id=zone.id, **attrs)