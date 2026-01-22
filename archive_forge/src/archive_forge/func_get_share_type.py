from heat.common import exception as heat_exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
from manilaclient import client as manila_client
from manilaclient import exceptions
from oslo_config import cfg
def get_share_type(self, share_type_identity):
    return self._find_resource_by_id_or_name(share_type_identity, self.client().share_types.list(), 'share type')