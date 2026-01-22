from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def get_filesystem_for_device(self, device):
    real_device = os.path.realpath(device)
    self.__check_init()
    for fs in self.__filesystems.values():
        if fs.contains_device(real_device):
            return fs
    return None