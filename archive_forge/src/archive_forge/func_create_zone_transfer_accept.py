from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def create_zone_transfer_accept(self, **attrs):
    """Create a new ZoneTransfer Accept from attributes

        :param dict attrs: Keyword arguments which will be used to create
            a :class:`~openstack.dns.v2.zone_transfer.ZoneTransferAccept`,
            comprised of the properties on the ZoneTransferAccept class.
        :returns: The results of zone transfer request creation.
        :rtype: :class:`~openstack.dns.v2.zone_transfer.ZoneTransferAccept`
        """
    return self._create(_zone_transfer.ZoneTransferAccept, **attrs)