from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def get_playbook_params(self, opt):
    return self.module.params[opt]