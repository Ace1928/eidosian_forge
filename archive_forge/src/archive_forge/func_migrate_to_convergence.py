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
def migrate_to_convergence(self):
    db_rsrcs = self.db_active_resources_get()
    res_id_dep = self.dependencies.translate(lambda res: res.id)
    current_template_id = self.t.id
    if db_rsrcs is not None:
        for db_res in db_rsrcs.values():
            requires = set(res_id_dep.requires(db_res.id))
            r = self.resources.get(db_res.name)
            if r is None:
                LOG.warning('Resource %(res)s not found in template for stack %(st)s, deleting from DB.', {'res': db_res.name, 'st': self.id})
                resource_objects.Resource.delete(self.context, db_res.id)
            else:
                r.requires = requires
                db_res.convert_to_convergence(current_template_id, requires)
    self.current_traversal = uuidutils.generate_uuid()
    self.convergence = True
    prev_raw_template_id = self.prev_raw_template_id
    self.prev_raw_template_id = None
    self.store(ignore_traversal_check=True)
    if prev_raw_template_id:
        raw_template_object.RawTemplate.delete(self.context, prev_raw_template_id)