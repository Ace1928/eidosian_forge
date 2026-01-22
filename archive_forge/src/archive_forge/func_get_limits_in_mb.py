from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
import copy
def get_limits_in_mb(limits):
    """
    :param limits: Limits in KB
    :return: Limits in MB
    """
    if limits:
        return limits / 1024