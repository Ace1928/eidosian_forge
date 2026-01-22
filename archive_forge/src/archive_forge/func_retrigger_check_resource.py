import eventlet.queue
import functools
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common import exception
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import sync_point
from heat.objects import resource as resource_objects
from heat.rpc import api as rpc_api
from heat.rpc import listener_client
def retrigger_check_resource(self, cnxt, resource_id, stack):
    current_traversal = stack.current_traversal
    graph = stack.convergence_dependencies.graph()
    update_key = parser.ConvergenceNode(resource_id, True)
    key = parser.ConvergenceNode(resource_id, update_key in graph)
    LOG.info('Re-trigger resource: %s', key)
    predecessors = set(graph[key])
    try:
        propagate_check_resource(cnxt, self._rpc_client, resource_id, current_traversal, predecessors, key, None, key.is_update, None)
    except exception.EntityNotFound as e:
        if e.entity != 'Sync Point':
            raise