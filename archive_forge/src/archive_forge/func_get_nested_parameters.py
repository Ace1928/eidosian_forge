import collections
import contextlib
import copy
import eventlet
import functools
import re
import warnings
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import timeutils as oslo_timeutils
from oslo_utils import uuidutils
from osprofiler import profiler
from heat.common import context as common_context
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import lifecycle_plugin_utils
from heat.engine import api
from heat.engine import dependencies
from heat.engine import environment
from heat.engine import event
from heat.engine.notification import stack as notification
from heat.engine import parameter_groups as param_groups
from heat.engine import parent_rsrc
from heat.engine import resource
from heat.engine import resources
from heat.engine import scheduler
from heat.engine import status
from heat.engine import stk_defn
from heat.engine import sync_point
from heat.engine import template as tmpl
from heat.engine import update
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import snapshot as snapshot_object
from heat.objects import stack as stack_object
from heat.objects import stack_tag as stack_tag_object
from heat.objects import user_creds as ucreds_object
from heat.rpc import api as rpc_api
from heat.rpc import worker_client as rpc_worker_client
def get_nested_parameters(self, filter_func):
    """Return nested parameters schema, if any.

        This introspects the resources to return the parameters of the nested
        stacks. It uses the `get_nested_parameters_stack` API to build the
        stack.
        """
    result = {}
    for name, rsrc in self.resources.items():
        nested = rsrc.get_nested_parameters_stack()
        if nested is None:
            continue
        nested_params = nested.parameters.map(api.format_validate_parameter, filter_func=filter_func)
        params = {'Type': rsrc.type(), 'Description': nested.t.get('Description', ''), 'Parameters': nested_params}
        nested_pg = param_groups.ParameterGroups(nested.t)
        if nested_pg.parameter_groups:
            params.update({'ParameterGroups': nested_pg.parameter_groups})
        params.update(nested.get_nested_parameters(filter_func))
        result[name] = params
    return {'NestedParameters': result} if result else {}