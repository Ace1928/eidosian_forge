from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.neutron import neutron
@staticmethod
def router_for_vpc(client, network_id):
    net = VPC.network_for_vpc(client, network_id)
    routers = client.list_routers(name=net['name'])['routers']
    if len(routers) == 0:
        return None
    if len(routers) > 1:
        raise exception.Error(_('Multiple routers found with name %s') % net['name'])
    return routers[0]