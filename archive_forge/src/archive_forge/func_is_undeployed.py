from __future__ import absolute_import, division, print_function
import os
import time
from ansible.module_utils.basic import AnsibleModule
def is_undeployed(deploy_path, deployment):
    return os.path.exists(os.path.join(deploy_path, '%s.undeployed' % deployment))