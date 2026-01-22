import collections
from oslo_log import log as logging
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import param_utils
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.engine import constraints as constr
from heat.rpc import api as rpc_api
def format_resource_properties(resource):

    def get_property(prop):
        try:
            return resource.properties[prop]
        except (KeyError, ValueError):
            LOG.exception('Error in fetching property %s of resource %s' % (prop, resource.name))
            return None
    return dict(((prop, get_property(prop)) for prop in resource.properties_schema.keys()))