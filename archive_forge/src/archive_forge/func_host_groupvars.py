from __future__ import (absolute_import, division, print_function)
import os
from ansible import constants as C
from ansible.errors import AnsibleParserError, AnsibleOptionsError
from ansible.inventory.helpers import get_group_vars
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.module_utils.common.text.converters import to_native
from ansible.utils.vars import combine_vars
from ansible.vars.fact_cache import FactCache
from ansible.vars.plugins import get_vars_from_inventory_sources
def host_groupvars(self, host, loader, sources):
    """ requires host object """
    gvars = get_group_vars(host.get_groups())
    if self.get_option('use_vars_plugins'):
        gvars = combine_vars(gvars, get_vars_from_inventory_sources(loader, sources, host.get_groups(), 'all'))
    return gvars