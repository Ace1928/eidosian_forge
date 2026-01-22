from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def get_rubygems_environ(module):
    if module.params['install_dir']:
        return {'GEM_HOME': module.params['install_dir']}
    return None