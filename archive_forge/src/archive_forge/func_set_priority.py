from __future__ import (absolute_import, division, print_function)
from collections.abc import Mapping, MutableMapping
from enum import Enum
from itertools import chain
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars
def set_priority(self, priority):
    try:
        self.priority = int(priority)
    except TypeError:
        pass