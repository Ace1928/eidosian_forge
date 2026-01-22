from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as nc
from oslo_config import cfg
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
def update_ext_resource(self, resource, prop_diff, resource_id):
    """Returns updated ext resource record."""
    path = self._resolve_resource_path(resource)
    return self.client().update_ext(path + '/%s', resource_id, {resource: prop_diff})