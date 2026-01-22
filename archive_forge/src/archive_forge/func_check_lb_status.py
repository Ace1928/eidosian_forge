from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as nc
from oslo_config import cfg
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
def check_lb_status(self, lb_id):
    lb = self.client().show_loadbalancer(lb_id)['loadbalancer']
    status = lb['provisioning_status']
    if status == 'ERROR':
        raise exception.ResourceInError(resource_status=status)
    return status == 'ACTIVE'