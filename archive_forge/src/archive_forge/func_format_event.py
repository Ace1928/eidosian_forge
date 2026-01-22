import itertools
from webob import exc
from heat.api.openstack.v1 import util
from heat.common.i18n import _
from heat.common import identifier
from heat.common import param_utils
from heat.common import serializers
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
def format_event(req, event, keys=None):

    def include_key(k):
        return k in keys if keys else True

    def transform(key, value):
        if not include_key(key):
            return
        if key == rpc_api.EVENT_ID:
            identity = identifier.EventIdentifier(**value)
            yield ('id', identity.event_id)
            yield ('links', [util.make_link(req, identity), util.make_link(req, identity.resource(), 'resource'), util.make_link(req, identity.stack(), 'stack')])
        elif key in (rpc_api.EVENT_STACK_ID, rpc_api.EVENT_STACK_NAME, rpc_api.EVENT_RES_ACTION):
            return
        elif key == rpc_api.EVENT_RES_STATUS and rpc_api.EVENT_RES_ACTION in event:
            yield (key, '_'.join((event[rpc_api.EVENT_RES_ACTION], value)))
        elif key == rpc_api.RES_NAME:
            yield ('logical_resource_id', value)
            yield (key, value)
        else:
            yield (key, value)
    ev = dict(itertools.chain.from_iterable((transform(k, v) for k, v in event.items())))
    root_stack_id = event.get(rpc_api.EVENT_ROOT_STACK_ID)
    if root_stack_id:
        root_identifier = identifier.HeatIdentifier(**root_stack_id)
        ev['links'].append(util.make_link(req, root_identifier, 'root_stack'))
    return ev