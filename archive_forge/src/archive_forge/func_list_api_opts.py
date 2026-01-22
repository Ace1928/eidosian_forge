import copy
import itertools
from osprofiler import opts as profiler
import glance.api.middleware.context
import glance.api.versions
import glance.async_.flows._internal_plugins
import glance.async_.flows.api_image_import
import glance.async_.flows.convert
from glance.async_.flows.plugins import plugin_opts
import glance.async_.taskflow_executor
import glance.common.config
import glance.common.location_strategy
import glance.common.location_strategy.store_type
import glance.common.property_utils
import glance.common.wsgi
import glance.image_cache
import glance.image_cache.drivers.sqlite
import glance.notifier
import glance.scrubber
def list_api_opts():
    """Return a list of oslo_config options available in Glance API service.

    Each element of the list is a tuple. The first element is the name of the
    group under which the list of elements in the second element will be
    registered. A group name of None corresponds to the [DEFAULT] group in
    config files.

    This function is also discoverable via the 'glance.api' entry point
    under the 'oslo_config.opts' namespace.

    The purpose of this is to allow tools like the Oslo sample config file
    generator to discover the options exposed to users by Glance.

    :returns: a list of (group_name, opts) tuples
    """
    return [(g, copy.deepcopy(o)) for g, o in _api_opts]