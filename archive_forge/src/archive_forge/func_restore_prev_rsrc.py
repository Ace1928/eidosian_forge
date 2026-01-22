from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.neutron import neutron
from heat.engine.resources.openstack.neutron import subnet
from heat.engine import support
from heat.engine import translation
def restore_prev_rsrc(self, convergence=False):
    if convergence:
        prev_port = self
        existing_port, rsrc_owning_stack, stack = resource.Resource.load(prev_port.context, prev_port.replaced_by, prev_port.stack.current_traversal, True, prev_port.stack.defn._resource_data)
        existing_port_id = existing_port.resource_id
    else:
        backup_stack = self.stack._backup_stack()
        prev_port = backup_stack.resources.get(self.name)
        existing_port_id = self.resource_id
    if existing_port_id:
        props = {'fixed_ips': []}
        self.client().update_port(existing_port_id, {'port': props})
    fixed_ips = prev_port.data().get('port_fip', [])
    if fixed_ips and prev_port.resource_id:
        prev_port_props = {'fixed_ips': jsonutils.loads(fixed_ips)}
        self.client().update_port(prev_port.resource_id, {'port': prev_port_props})