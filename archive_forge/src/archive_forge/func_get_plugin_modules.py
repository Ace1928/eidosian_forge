import importlib
import logging
import sys
from osc_lib import clientmanager
from osc_lib import shell
import stevedore
def get_plugin_modules(group):
    """Find plugin entry points"""
    mod_list = []
    mgr = stevedore.ExtensionManager(group)
    for ep in mgr:
        LOG.debug('Found plugin %s', ep.name)
        try:
            module_name = ep.entry_point.module_name
        except AttributeError:
            try:
                module_name = ep.entry_point.module
            except AttributeError:
                module_name = ep.entry_point.value
        try:
            module = importlib.import_module(module_name)
        except Exception as err:
            sys.stderr.write('WARNING: Failed to import plugin %s: %s.\n' % (ep.name, err))
            continue
        mod_list.append(module)
        init_func = getattr(module, 'Initialize', None)
        if init_func:
            init_func('x')
        setattr(clientmanager.ClientManager, module.API_NAME, clientmanager.ClientCache(getattr(sys.modules[module_name], 'make_client', None)))
    return mod_list