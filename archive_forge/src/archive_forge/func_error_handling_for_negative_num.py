from __future__ import (absolute_import, division, print_function)
import os
import tempfile
import copy
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
def error_handling_for_negative_num(option, val):
    return '{0} cannot be a negative number or zero,got {1}'.format(option, val)