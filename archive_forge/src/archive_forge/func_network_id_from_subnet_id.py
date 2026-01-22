from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as nc
from oslo_config import cfg
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
def network_id_from_subnet_id(self, subnet_id):
    subnet_info = self.client().show_subnet(subnet_id)
    return subnet_info['subnet']['network_id']