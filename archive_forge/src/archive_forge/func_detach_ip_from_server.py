import ipaddress
import time
import warnings
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
from openstack import warnings as os_warnings
def detach_ip_from_server(self, server_id, floating_ip_id):
    """Detach a floating IP from a server.

        :param server_id: ID of a server.
        :param floating_ip_id: Id of the floating IP to detach.

        :returns: True if the IP has been detached, or False if the IP wasn't
            attached to any server.
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    if self._use_neutron_floating():
        try:
            return self._neutron_detach_ip_from_server(server_id=server_id, floating_ip_id=floating_ip_id)
        except exceptions.NotFoundException as e:
            self.log.debug("Something went wrong talking to neutron API: '%(msg)s'. Trying with Nova.", {'msg': str(e)})
    self._nova_detach_ip_from_server(server_id=server_id, floating_ip_id=floating_ip_id)