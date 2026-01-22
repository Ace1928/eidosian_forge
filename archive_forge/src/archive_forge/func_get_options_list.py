import argparse
from keystoneauth1.identity.v3 import k2k
from keystoneauth1.loading import base
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from osc_lib import utils
def get_options_list():
    """Gather plugin options so the help action has them available"""
    global OPTIONS_LIST
    if not OPTIONS_LIST:
        for plugin_name in get_plugin_list():
            plugin_options = base.get_plugin_options(plugin_name)
            for o in plugin_options:
                os_name = o.name.lower().replace('_', '-')
                os_env_name = 'OS_' + os_name.upper().replace('-', '_')
                OPTIONS_LIST.setdefault(os_name, {'env': os_env_name, 'help': ''})
                OPTIONS_LIST[os_name]['help'] += 'With %s: %s\n' % (plugin_name, o.help)
    return OPTIONS_LIST