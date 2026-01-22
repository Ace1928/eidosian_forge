import argparse
import os
from keystoneauth1.loading import base
def load_from_argparse_arguments(namespace, **kwargs):
    """Retrieve the created plugin from the completed argparse results.

    Loads and creates the auth plugin from the information parsed from the
    command line by argparse.

    :param Namespace namespace: The result from CLI parsing.

    :returns: An auth plugin, or None if a name is not provided.
    :rtype: :class:`keystoneauth1.plugin.BaseAuthPlugin`

    :raises keystoneauth1.exceptions.auth_plugins.NoMatchingPlugin:
        if a plugin cannot be created.
    """
    if not namespace.os_auth_type:
        return None
    if isinstance(namespace.os_auth_type, type):
        plugin = namespace.os_auth_type
    else:
        plugin = base.get_plugin_loader(namespace.os_auth_type)

    def _getter(opt):
        return getattr(namespace, 'os_%s' % opt.dest)
    return plugin.load_from_options_getter(_getter, **kwargs)