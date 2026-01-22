import eventlet
from oslo_config import cfg
import oslo_messaging
from oslo_messaging.rpc import dispatcher
from osprofiler import profiler
from heat.common import context
def serialize_entity(self, ctxt, entity):
    if not self._base:
        return entity
    return self._base.serialize_entity(ctxt, entity)