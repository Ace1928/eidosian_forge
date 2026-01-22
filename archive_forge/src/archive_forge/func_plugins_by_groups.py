from __future__ import (absolute_import, division, print_function)
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, MutableMapping, Sequence
from hashlib import sha1
from jinja2.exceptions import UndefinedError
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleParserError, AnsibleUndefinedVariable, AnsibleFileNotFound, AnsibleAssertionError, AnsibleTemplateError
from ansible.inventory.host import Host
from ansible.inventory.helpers import sort_groups, get_group_vars
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import text_type, string_types
from ansible.plugins.loader import lookup_loader
from ansible.vars.fact_cache import FactCache
from ansible.template import Templar
from ansible.utils.display import Display
from ansible.utils.listify import listify_lookup_plugin_terms
from ansible.utils.vars import combine_vars, load_extra_vars, load_options_vars
from ansible.utils.unsafe_proxy import wrap_var
from ansible.vars.clean import namespace_facts, clean_facts
from ansible.vars.plugins import get_vars_from_inventory_sources, get_vars_from_path
def plugins_by_groups():
    """
                    merges all plugin sources by group,
                    This should be used instead, NOT in combination with the other groups_plugins* functions
                """
    data = {}
    for group in host_groups:
        data[group] = _combine_and_track(data[group], _plugins_inventory(group), "inventory group_vars for '%s'" % group)
        data[group] = _combine_and_track(data[group], _plugins_play(group), "playbook group_vars for '%s'" % group)
    return data