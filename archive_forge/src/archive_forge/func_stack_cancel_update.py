import collections
import copy
import datetime
import functools
import itertools
import os
import pydoc
import signal
import socket
import sys
import eventlet
from oslo_config import cfg
from oslo_context import context as oslo_context
from oslo_log import log as logging
import oslo_messaging as messaging
from oslo_serialization import jsonutils
from oslo_service import service
from oslo_service import threadgroup
from oslo_utils import timeutils
from oslo_utils import uuidutils
from osprofiler import profiler
import webob
from heat.common import context
from heat.common import environment_format as env_fmt
from heat.common import environment_util as env_util
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import messaging as rpc_messaging
from heat.common import policy
from heat.common import service_utils
from heat.engine import api
from heat.engine import attributes
from heat.engine.cfn import template as cfntemplate
from heat.engine import clients
from heat.engine import environment
from heat.engine.hot import functions as hot_functions
from heat.engine import parameter_groups
from heat.engine import properties
from heat.engine import resources
from heat.engine import service_software_config
from heat.engine import stack as parser
from heat.engine import stack_lock
from heat.engine import stk_defn
from heat.engine import support
from heat.engine import template as templatem
from heat.engine import template_files
from heat.engine import update
from heat.engine import worker
from heat.objects import event as event_object
from heat.objects import resource as resource_objects
from heat.objects import service as service_objects
from heat.objects import snapshot as snapshot_object
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.rpc import worker_api as rpc_worker_api
@context.request_context
def stack_cancel_update(self, cnxt, stack_identity, cancel_with_rollback=True):
    """Cancel currently running stack update.

        :param cnxt: RPC context.
        :param stack_identity: Name of the stack for which to cancel update.
        :param cancel_with_rollback: Force rollback when cancel update.
        """
    db_stack = self._get_stack(cnxt, stack_identity)
    current_stack = parser.Stack.load(cnxt, stack=db_stack)
    if cancel_with_rollback:
        allowed_actions = (current_stack.UPDATE,)
    else:
        allowed_actions = (current_stack.UPDATE, current_stack.CREATE)
    if not (current_stack.status == current_stack.IN_PROGRESS and current_stack.action in allowed_actions):
        state = '_'.join(current_stack.state)
        msg = _('Cancelling update when stack is %s') % str(state)
        raise exception.NotSupported(feature=msg)
    LOG.info('Starting cancel of updating stack %s', db_stack.name)
    if current_stack.convergence:
        current_stack.thread_group_mgr = self.thread_group_mgr
        if cancel_with_rollback:
            func = current_stack.rollback
        else:
            func = functools.partial(self.worker_service.stop_traversal, current_stack)
        self.thread_group_mgr.start(current_stack.id, func)
        return
    lock = stack_lock.StackLock(cnxt, current_stack.id, self.engine_id)
    engine_id = lock.get_engine_id()
    if engine_id is None:
        LOG.debug('No lock found on stack %s', db_stack.name)
        return
    if cancel_with_rollback:
        cancel_message = rpc_api.THREAD_CANCEL_WITH_ROLLBACK
    else:
        cancel_message = rpc_api.THREAD_CANCEL
    if engine_id == self.engine_id:
        self.thread_group_mgr.send(current_stack.id, cancel_message)
    elif service_utils.engine_alive(cnxt, engine_id):
        cancel_result = self._remote_call(cnxt, engine_id, cfg.CONF.engine_life_check_timeout, self.listener.SEND, stack_identity=stack_identity, message=cancel_message)
        if cancel_result is None:
            LOG.debug('Successfully sent %(msg)s message to remote task on engine %(eng)s' % {'eng': engine_id, 'msg': cancel_message})
        else:
            raise exception.EventSendFailed(stack_name=current_stack.name, engine_id=engine_id)
    else:
        LOG.warning(_('Cannot cancel stack %(stack_name)s: lock held by unknown engine %(engine_id)s') % {'stack_name': db_stack.name, 'engine_id': engine_id})