from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def get_mountpath_as_child(self, subvolume_name):
    """Find a path to the target subvolume through a mounted ancestor"""
    nearest = self.get_nearest_subvolume(subvolume_name)
    if nearest.path == subvolume_name:
        nearest = nearest.get_parent_subvolume()
    if nearest is None or nearest.get_mounted_path() is None:
        raise BtrfsModuleException("Failed to find a path '%s' through a mounted parent subvolume" % subvolume_name)
    else:
        return nearest.get_mounted_path() + os.path.sep + nearest.get_child_relative_path(subvolume_name)