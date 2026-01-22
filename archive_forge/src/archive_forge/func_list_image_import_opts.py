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
def list_image_import_opts():
    """Return a list of oslo_config options available for Image Import"""
    opts = copy.deepcopy(_image_import_opts)
    opts.extend(plugin_opts.get_plugin_opts())
    return [(g, copy.deepcopy(o)) for g, o in opts]