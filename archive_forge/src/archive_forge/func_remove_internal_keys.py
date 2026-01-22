from __future__ import (absolute_import, division, print_function)
import os
import re
from collections.abc import MutableMapping, MutableSequence
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils import six
from ansible.plugins.loader import connection_loader
from ansible.utils.display import Display
def remove_internal_keys(data):
    """
    More nuanced version of strip_internal_keys
    """
    for key in list(data.keys()):
        if key.startswith('_ansible_') and key != '_ansible_parsed' or key in C.INTERNAL_RESULT_KEYS:
            display.warning('Removed unexpected internal key in module return: %s = %s' % (key, data[key]))
            del data[key]
    for key in ['warnings', 'deprecations']:
        if key in data and (not data[key]):
            del data[key]
    for key in list(data.get('ansible_facts', {}).keys()):
        if key.startswith('discovered_interpreter_') or key.startswith('ansible_discovered_interpreter_'):
            del data['ansible_facts'][key]