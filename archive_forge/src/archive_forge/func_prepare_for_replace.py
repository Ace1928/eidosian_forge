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
def prepare_for_replace(self):
    if self.resource_id is None:
        return
    with self.client_plugin().ignore_not_found:
        fixed_ips = self._show_resource().get('fixed_ips', [])
        self.data_set('port_fip', jsonutils.dumps(fixed_ips))
        props = {'fixed_ips': []}
        self.client().update_port(self.resource_id, {'port': props})