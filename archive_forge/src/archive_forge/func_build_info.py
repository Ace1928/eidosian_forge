from oslo_config import cfg
from heat.api.openstack.v1 import util
from heat.common import serializers
from heat.common import wsgi
from heat.rpc import client as rpc_client
@util.registered_policy_enforce
def build_info(self, req):
    engine_revision = self.rpc_client.get_revision(req.context)
    build_info = {'api': {'revision': cfg.CONF.revision['heat_revision']}, 'engine': {'revision': engine_revision}}
    return build_info