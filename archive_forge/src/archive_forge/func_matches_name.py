from __future__ import (absolute_import, division, print_function)
from abc import ABC
import types
import typing as t
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible.utils.display import Display
def matches_name(self, possible_names):
    possible_fqcns = set()
    for name in possible_names:
        if '.' not in name:
            possible_fqcns.add(f'ansible.builtin.{name}')
        elif name.startswith('ansible.legacy.'):
            possible_fqcns.add(name.removeprefix('ansible.legacy.'))
        possible_fqcns.add(name)
    return bool(possible_fqcns.intersection(set(self.ansible_aliases)))