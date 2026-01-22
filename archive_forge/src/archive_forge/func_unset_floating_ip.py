from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def unset_floating_ip(self, floating_ip):
    """Unset a Floating IP PTR record
        :param floating_ip: ID for the floatingip associated with the
            project.
        :returns: FloatingIP PTR record.
        :rtype: :class:`~openstack.dns.v2.fip.FloatipgIP`
        """
    attrs = {'ptrdname': None}
    return self._update(_fip.FloatingIP, floating_ip, **attrs)