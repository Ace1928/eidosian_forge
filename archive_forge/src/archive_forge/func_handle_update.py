from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
def handle_update(self, json_snippet, tmpl_diff, prop_diff):
    if prop_diff:
        self.prepare_update_properties(prop_diff)
        self.client_plugin().update_ext_resource('tap_flow', prop_diff, self.resource_id)