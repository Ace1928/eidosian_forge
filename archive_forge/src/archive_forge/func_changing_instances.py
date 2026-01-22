import functools
from oslo_log import log as logging
from heat.common import environment_format
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.common import short_id
from heat.common import timeutils as iso8601utils
from heat.engine import attributes
from heat.engine import environment
from heat.engine import output
from heat.engine import properties
from heat.engine.resources import stack_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.scaling import lbutils
from heat.scaling import rolling_update
from heat.scaling import template
def changing_instances(old_tmpl, new_tmpl):
    updated = set(new_tmpl.resource_definitions(None).items())
    if old_tmpl is not None:
        current = set(old_tmpl.resource_definitions(None).items())
        changing = current ^ updated
    else:
        changing = updated
    return set((k for k, v in changing))