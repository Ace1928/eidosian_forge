from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def get_mountpoints(self):
    return self.__filesystem.get_mountpoints_by_subvolume_id(self.__subvolume_id)