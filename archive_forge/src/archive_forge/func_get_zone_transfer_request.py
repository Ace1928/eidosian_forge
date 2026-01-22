from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def get_zone_transfer_request(self, request):
    """Get a ZoneTransfer Request info

        :param request: The value can be the ID of a transfer request
            or a :class:`~openstack.dns.v2.zone_transfer.ZoneTransferRequest`
            instance.
        :returns: Zone transfer request instance.
        :rtype: :class:`~openstack.dns.v2.zone_transfer.ZoneTransferRequest`
        """
    return self._get(_zone_transfer.ZoneTransferRequest, request)