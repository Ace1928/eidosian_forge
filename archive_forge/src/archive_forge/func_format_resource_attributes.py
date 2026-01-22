import collections
from oslo_log import log as logging
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import param_utils
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.engine import constraints as constr
from heat.rpc import api as rpc_api
def format_resource_attributes(resource, with_attr=None):
    resolver = resource.attributes
    if not with_attr:
        with_attr = []
    resolver.reset_resolved_values()

    def resolve(attr, resolver):
        try:
            return resolver._resolver(attr)
        except Exception:
            return None
    if 'show' in resolver:
        show_attr = resolve('show', resolver)
        if isinstance(show_attr, collections.abc.Mapping):
            for a in with_attr:
                if a not in show_attr:
                    show_attr[a] = resolve(a, resolver)
            return show_attr
        else:
            del resolver._attributes['show']
    attributes = set(resolver) | set(with_attr)
    return dict(((attr, resolve(attr, resolver)) for attr in attributes))