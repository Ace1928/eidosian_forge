from __future__ import (absolute_import, division, print_function)
from abc import ABC
import types
import typing as t
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible.utils.display import Display
@property
def option_definitions(self):
    if self._defs is None:
        self._defs = C.config.get_configuration_definitions(plugin_type=self.plugin_type, name=self._load_name)
    return self._defs