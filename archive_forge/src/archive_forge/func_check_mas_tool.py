from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import os
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
import platform
def check_mas_tool(self):
    """ Verifies that the `mas` tool is available in a recent version """
    if not self.mas_path:
        self.module.fail_json(msg='Required `mas` tool is not installed')
    rc, out, err = self.run(['version'])
    if rc != 0 or not out.strip() or LooseVersion(out.strip()) < LooseVersion('1.5.0'):
        self.module.fail_json(msg='`mas` tool in version 1.5.0+ needed, got ' + out.strip())