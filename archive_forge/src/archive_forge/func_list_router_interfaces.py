from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
def list_router_interfaces(self, router, interface_type=None):
    """List all interfaces for a router.

        :param dict router: A router dict object.
        :param string interface_type: One of None, "internal", or "external".
            Controls whether all, internal interfaces or external interfaces
            are returned.
        :returns: A list of network ``Port`` objects.
        """
    ports = list(self.network.ports(device_id=router['id']))
    router_interfaces = [port for port in ports if port['device_owner'] in ['network:router_interface', 'network:router_interface_distributed', 'network:ha_router_replicated_interface']] if not interface_type or interface_type == 'internal' else []
    router_gateways = [port for port in ports if port['device_owner'] == 'network:router_gateway'] if not interface_type or interface_type == 'external' else []
    return router_interfaces + router_gateways