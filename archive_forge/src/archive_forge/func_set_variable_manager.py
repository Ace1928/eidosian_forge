from __future__ import (absolute_import, division, print_function)
from collections.abc import Mapping
from ansible import constants as C
from ansible.template import Templar, AnsibleUndefined
def set_variable_manager(self, variable_manager):
    self._variable_manager = variable_manager
    variable_manager._hostvars = self