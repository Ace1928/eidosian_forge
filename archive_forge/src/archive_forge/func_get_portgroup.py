from ironicclient.common.apiclient import exceptions as ic_exc
from ironicclient.v1 import client as ironic_client
from oslo_config import cfg
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
def get_portgroup(self, value):
    return self._get_rsrc_name_or_id(value, entity='portgroup', entity_msg='PortGroup')