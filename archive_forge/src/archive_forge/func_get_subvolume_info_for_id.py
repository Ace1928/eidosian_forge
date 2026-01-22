from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def get_subvolume_info_for_id(self, subvolume_id):
    return self.__subvolumes[subvolume_id] if subvolume_id in self.__subvolumes else None