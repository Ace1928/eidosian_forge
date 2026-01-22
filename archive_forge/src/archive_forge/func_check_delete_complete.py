from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
def check_delete_complete(self, data):
    if self.resource_id is None:
        return True
    with self.client_plugin().ignore_not_found:
        try:
            if self.client_plugin().check_ext_resource_status('tap_flow', self.resource_id):
                self.client_plugin().delete_ext_resource('tap_flow', self.resource_id)
        except exception.ResourceInError:
            self.client_plugin().delete_ext_resource('tap_flow', self.resource_id)
        return False
    return True