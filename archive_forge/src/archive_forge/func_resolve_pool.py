from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as nc
from oslo_config import cfg
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
def resolve_pool(self, props, pool_key, pool_id_key):
    if props.get(pool_key):
        props[pool_id_key] = self.find_resourceid_by_name_or_id('pool', props.get(pool_key))
        props.pop(pool_key)
    return props[pool_id_key]