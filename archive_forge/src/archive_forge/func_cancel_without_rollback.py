from webob import exc
from heat.api.openstack.v1 import util
from heat.common.i18n import _
from heat.common import serializers
from heat.common import wsgi
from heat.rpc import client as rpc_client
@util.registered_identified_stack
def cancel_without_rollback(self, req, identity, body=None):
    self.rpc_client.stack_cancel_update(req.context, identity, cancel_with_rollback=False)