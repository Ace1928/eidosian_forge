from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def contains_subvolume(self, subvolume):
    return self.get_subvolume_by_name(subvolume) is not None