from __future__ import (absolute_import, division, print_function)
import os
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
Boot to a network ISO image