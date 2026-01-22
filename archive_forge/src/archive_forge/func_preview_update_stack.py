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
def preview_update_stack(self, cnxt, stack_identity, template, params, files, args, environment_files=None, files_container=None):
    """Shows the resources that would be updated.

        The preview_update_stack method shows the resources that would be
        changed with an update to an existing stack based on the provided
        template and parameters. See update_stack for description of
        parameters.

        This method *cannot* guarantee that an update will have the actions
        specified because resource plugins can influence changes/replacements
        at runtime.

        Note that at this stage the template has already been fetched from the
        heat-api process if using a template-url.
        """
    db_stack = self._get_stack(cnxt, stack_identity)
    LOG.info('Previewing update of stack %s', db_stack.name)
    current_stack = parser.Stack.load(cnxt, stack=db_stack)
    tmpl, current_stack, updated_stack = self._prepare_stack_updates(cnxt, current_stack, template, params, environment_files, files, files_container, args)
    update_task = update.StackUpdate(current_stack, updated_stack, None)
    actions = update_task.preview()

    def fmt_action_map(current, updated, act):

        def fmt_updated_res(k):
            return api.format_stack_resource(updated.resources.get(k))

        def fmt_current_res(k):
            return api.format_stack_resource(current.resources.get(k))
        return {'unchanged': list(map(fmt_updated_res, act.get('unchanged', []))), 'updated': list(map(fmt_current_res, act.get('updated', []))), 'replaced': list(map(fmt_updated_res, act.get('replaced', []))), 'added': list(map(fmt_updated_res, act.get('added', []))), 'deleted': list(map(fmt_current_res, act.get('deleted', [])))}
    updated_stack.id = current_stack.id
    fmt_actions = fmt_action_map(current_stack, updated_stack, actions)
    if args.get(rpc_api.PARAM_SHOW_NESTED):
        updated_stack.preview_resources()

        def nested_fmt_actions(current, updated, act):
            updated.id = current.id

            def _n_deleted(stk, deleted):
                for rsrc in deleted:
                    deleted_rsrc = stk.resources.get(rsrc)
                    if deleted_rsrc.has_nested():
                        nested_stk = deleted_rsrc.nested()
                        nested_rsrc = nested_stk.resources.keys()
                        n_fmt = fmt_action_map(nested_stk, None, {'deleted': nested_rsrc})
                        fmt_actions['deleted'].extend(n_fmt['deleted'])
                        _n_deleted(nested_stk, nested_rsrc)
            _n_deleted(current, act['deleted'] + act['replaced'])

            def _n_added(stk, added):
                for rsrc in added:
                    added_rsrc = stk.resources.get(rsrc)
                    if added_rsrc.has_nested():
                        nested_stk = added_rsrc.nested()
                        nested_rsrc = nested_stk.resources.keys()
                        n_fmt = fmt_action_map(None, nested_stk, {'added': nested_rsrc})
                        fmt_actions['added'].extend(n_fmt['added'])
                        _n_added(nested_stk, nested_rsrc)
            _n_added(updated, act['added'] + act['replaced'])
            for rsrc in act['updated']:
                current_rsrc = current.resources.get(rsrc)
                updated_rsrc = updated.resources.get(rsrc)
                if current_rsrc.has_nested() and updated_rsrc.has_nested():
                    current_nested = current_rsrc.nested()
                    updated_nested = updated_rsrc.nested()
                    update_task = update.StackUpdate(current_nested, updated_nested, None)
                    n_actions = update_task.preview()
                    n_fmt_actions = fmt_action_map(current_nested, updated_nested, n_actions)
                    for k in fmt_actions:
                        fmt_actions[k].extend(n_fmt_actions[k])
                    nested_fmt_actions(current_nested, updated_nested, n_actions)
        nested_fmt_actions(current_stack, updated_stack, actions)
    return fmt_actions