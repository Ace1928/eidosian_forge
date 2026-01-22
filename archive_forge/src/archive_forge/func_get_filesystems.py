from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def get_filesystems(self):
    self.__check_init()
    return list(self.__filesystems.values())