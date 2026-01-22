from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from datetime import datetime, timedelta
import time
import copy
def calculate_retention(desired_retention=None, retention_unit=None):
    """
    :param desired_retention: Desired retention of the snapshot
    :param retention_unit: Retention unit for snapshot
    :return: Retention in minutes
    """
    retention = 0
    if retention_unit == 'days':
        retention = desired_retention * 24 * 60
    else:
        retention = desired_retention * 60
    return retention