from datetime import datetime
from libcloud.common.gandi import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_interfaces(self):
    """
        Specific method to list network interfaces

        :rtype: ``list`` of :class:`GandiNetworkInterface`
        """
    ifaces = self.connection.request('hosting.iface.list').object
    ips = self.connection.request('hosting.ip.list').object
    for iface in ifaces:
        iface['ips'] = list(filter(lambda i: i['iface_id'] == iface['id'], ips))
    return self._to_ifaces(ifaces)