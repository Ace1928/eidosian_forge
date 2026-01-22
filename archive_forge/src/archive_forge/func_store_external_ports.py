import itertools
import eventlet
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import netutils
import tenacity
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
from heat.engine.resources.openstack.neutron import port as neutron_port
def store_external_ports(self):
    """Store in resource's data IDs of ports created by nova for server.

        If no port property is specified and no internal port has been created,
        nova client takes no port-id and calls port creating into server
        creating. We need to store information about that ports, so store
        their IDs to data with key `external_ports`.
        """
    server = self.client().servers.get(self.resource_id)
    ifaces = server.interface_list()
    external_port_ids = set((iface.port_id for iface in ifaces))
    data_external_port_ids = set((port['id'] for port in self._data_get_ports('external_ports')))
    for port_id in data_external_port_ids - external_port_ids:
        self._data_update_ports(port_id, 'delete', port_type='external_ports')
    internal_port_ids = set((port['id'] for port in self._data_get_ports()))
    new_ports = external_port_ids - internal_port_ids - data_external_port_ids
    for port_id in new_ports:
        self._data_update_ports(port_id, 'add', port_type='external_ports')