from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def get_parent_subvolume(self):
    parent_id = self.parent
    return self.__filesystem.get_subvolume_by_id(parent_id) if parent_id is not None else None