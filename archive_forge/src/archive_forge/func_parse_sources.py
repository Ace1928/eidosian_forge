from __future__ import (absolute_import, division, print_function)
import fnmatch
import os
import sys
import re
import itertools
import traceback
from operator import attrgetter
from random import shuffle
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleOptionsError, AnsibleParserError
from ansible.inventory.data import InventoryData
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.parsing.utils.addresses import parse_address
from ansible.plugins.loader import inventory_loader
from ansible.utils.helpers import deduplicate_list
from ansible.utils.path import unfrackpath
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars
from ansible.vars.plugins import get_vars_from_inventory_sources
def parse_sources(self, cache=False):
    """ iterate over inventory sources and parse each one to populate it"""
    parsed = False
    for source in self._sources:
        if source:
            if ',' not in source:
                source = unfrackpath(source, follow=False)
            parse = self.parse_source(source, cache=cache)
            if parse and (not parsed):
                parsed = True
    if parsed:
        self._inventory.reconcile_inventory()
    elif C.INVENTORY_UNPARSED_IS_FAILED:
        raise AnsibleError('No inventory was parsed, please check your configuration and options.')
    elif C.INVENTORY_UNPARSED_WARNING:
        display.warning('No inventory was parsed, only implicit localhost is available')
    for group in self.groups.values():
        group.vars = combine_vars(group.vars, get_vars_from_inventory_sources(self._loader, self._sources, [group], 'inventory'))
    for host in self.hosts.values():
        host.vars = combine_vars(host.vars, get_vars_from_inventory_sources(self._loader, self._sources, [host], 'inventory'))