import os
from debtcollector import removals
from keystoneauth1 import plugin
import stevedore
from keystoneclient import exceptions
@removals.remove(message='keystoneclient auth plugins are deprecated. Use keystoneauth.', version='2.1.0', removal_version='3.0.0')
def get_available_plugin_classes():
    """Retrieve all the plugin classes available on the system.

    :returns: A dict with plugin entrypoint name as the key and the plugin
              class as the value.
    :rtype: dict
    """
    mgr = stevedore.ExtensionManager(namespace=PLUGIN_NAMESPACE, propagate_map_exceptions=True, invoke_on_load=False)
    return dict(mgr.map(lambda ext: (ext.entry_point.name, ext.plugin)))