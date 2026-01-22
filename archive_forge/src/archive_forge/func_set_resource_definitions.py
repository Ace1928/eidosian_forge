from __future__ import (absolute_import, division, print_function)
import traceback
from abc import abstractmethod
from ansible.module_utils._text import to_native
def set_resource_definitions(self):
    self.resource_definitions = create_definitions(self.params)