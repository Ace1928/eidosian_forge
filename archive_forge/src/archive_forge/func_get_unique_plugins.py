import weakref
from oslo_concurrency import lockutils
from neutron_lib.plugins import constants
def get_unique_plugins():
    return _get_plugin_directory().unique_plugins