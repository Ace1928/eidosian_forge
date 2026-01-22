import collections
import functools
from oslo_utils import strutils
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import support
from oslo_log import log as logging
def get_cache_mode(self, attribute_name):
    """Return the cache mode for the specified attribute.

        If the attribute is not defined in the schema, the default cache
        mode (CACHE_LOCAL) is returned.
        """
    try:
        return self._attributes[attribute_name].schema.cache_mode
    except KeyError:
        return Schema.CACHE_LOCAL