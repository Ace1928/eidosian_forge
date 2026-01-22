from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as nc
from oslo_config import cfg
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
def is_not_found(self, ex):
    if isinstance(ex, (exceptions.NotFound, exceptions.NetworkNotFoundClient, exceptions.PortNotFoundClient)):
        return True
    return isinstance(ex, exceptions.NeutronClientException) and ex.status_code == 404