from __future__ import (absolute_import, division, print_function)
import keyword
import random
import uuid
from collections.abc import MutableMapping, MutableSequence
from json import dumps
from ansible import constants as C
from ansible import context
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.parsing.splitter import parse_kv
def load_options_vars(version):
    if not getattr(load_options_vars, 'options_vars', None):
        if version is None:
            version = 'Unknown'
        options_vars = {'ansible_version': version}
        attrs = {'check': 'check_mode', 'diff': 'diff_mode', 'forks': 'forks', 'inventory': 'inventory_sources', 'skip_tags': 'skip_tags', 'subset': 'limit', 'tags': 'run_tags', 'verbosity': 'verbosity'}
        for attr, alias in attrs.items():
            opt = context.CLIARGS.get(attr)
            if opt is not None:
                options_vars['ansible_%s' % alias] = opt
        setattr(load_options_vars, 'options_vars', options_vars)
    return load_options_vars.options_vars