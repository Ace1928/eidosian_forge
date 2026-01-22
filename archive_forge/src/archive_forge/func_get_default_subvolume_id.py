from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def get_default_subvolume_id(self, filesystem_path):
    return self.__btrfs_api.subvolume_get_default(filesystem_path)