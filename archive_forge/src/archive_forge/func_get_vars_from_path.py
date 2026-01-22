from __future__ import annotations
import os
from functools import lru_cache
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.inventory.group import InventoryObjectType
from ansible.plugins.loader import vars_loader
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars
def get_vars_from_path(loader, path, entities, stage):
    data = {}
    if vars_loader._paths is None:
        _prime_vars_loader()
    for plugin_name in vars_loader._plugin_instance_cache:
        if (plugin := vars_loader.get(plugin_name)) is None:
            continue
        collection = '.' in plugin.ansible_name and (not plugin.ansible_name.startswith('ansible.builtin.'))
        if collection and (hasattr(plugin, 'REQUIRES_ENABLED') or hasattr(plugin, 'REQUIRES_WHITELIST')):
            display.warning('Vars plugins in collections must be enabled to be loaded, REQUIRES_ENABLED is not supported. This should be removed from the plugin %s.' % plugin.ansible_name)
        if not _plugin_should_run(plugin, stage):
            continue
        if (new_vars := get_plugin_vars(loader, plugin, path, entities)) != {}:
            data = combine_vars(data, new_vars)
    return data