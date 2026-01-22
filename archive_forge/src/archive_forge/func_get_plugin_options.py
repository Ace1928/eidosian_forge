import abc
import stevedore
from keystoneauth1 import exceptions
def get_plugin_options(name):
    """Get the options for a specific plugin.

    This will be the list of options that is registered and loaded by the
    specified plugin.

    :returns: A list of :py:class:`keystoneauth1.loading.Opt` options.

    :raises keystoneauth1.exceptions.auth_plugins.NoMatchingPlugin:
        if a plugin cannot be created.
    """
    return get_plugin_loader(name).get_options()