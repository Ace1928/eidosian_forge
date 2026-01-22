from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import re
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def compare_chmod_value(self, current_permissions, desired_permissions):
    """
        compare current unix_permissions to desired unix_permissions.
        :return: True if the same, False it not the same or desired unix_permissions is not valid.
        """
    if current_permissions is None:
        return False
    if desired_permissions.isdigit():
        return int(current_permissions) == int(desired_permissions)
    if len(desired_permissions) not in [12, 9]:
        return False
    desired_octal_value = ''
    if len(desired_permissions) == 12:
        if desired_permissions[0] not in ['s', '-'] or desired_permissions[1] not in ['s', '-'] or desired_permissions[2] not in ['t', '-']:
            return False
        desired_octal_value += str(self.char_to_octal(desired_permissions[:3]))
    start_range = len(desired_permissions) - 9
    for i in range(start_range, len(desired_permissions), 3):
        if desired_permissions[i] not in ['r', '-'] or desired_permissions[i + 1] not in ['w', '-'] or desired_permissions[i + 2] not in ['x', '-']:
            return False
        group_permission = self.char_to_octal(desired_permissions[i:i + 3])
        desired_octal_value += str(group_permission)
    return int(current_permissions) == int(desired_octal_value)