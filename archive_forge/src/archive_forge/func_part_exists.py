from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import math
import re
import os
def part_exists(partitions, attribute, number):
    """
    Looks if a partition that has a specific value for a specific attribute
    actually exists.
    """
    return any((part[attribute] and part[attribute] == number for part in partitions))