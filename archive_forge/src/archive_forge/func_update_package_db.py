from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def update_package_db(module, xbps_path):
    """Returns True if update_package_db changed"""
    cmd = '%s -S' % xbps_path['install']
    rc, stdout, stderr = module.run_command(cmd, check_rc=False)
    if rc != 0:
        module.fail_json(msg='Could not update package db')
    if 'avg rate' in stdout:
        return True
    else:
        return False